// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use serde::Serialize;
use tauri::api::process::Command;

/// Kết quả tách audio cục bộ. Chỉ AUDIO rời máy người dùng — video thô KHÔNG bao
/// giờ được upload (quyền riêng tư + giảm ~600MB video xuống ~30MB audio).
#[derive(Serialize)]
struct AudioInfo {
    audio_path: String,
    md5: String,
    size_bytes: u64,
}

fn temp_path(stem: &str, ext: &str) -> String {
    // Tên file tạm duy nhất, không cần crate random: dùng mốc thời gian nano.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut dir = std::env::temp_dir();
    dir.push(format!("omnivoice_{}_{}.{}", stem, nanos, ext));
    dir.to_string_lossy().into_owned()
}

/// TR-2 (dọn rác temp trên đường LỖI): các command dưới đây tạo `out_path` rồi có thể
/// thoát sớm bằng `Err` (ffmpeg fail, đọc/ghi lỗi, file rỗng). Trên nhánh `Err`, `out_path`
/// KHÔNG được trả về frontend nên webview KHÔNG thể gọi `cleanup_temp_file` để dọn — file
/// rác (có thể là bản mp4/wav dở dang) nằm lại vĩnh viễn trong thư mục temp, phình đĩa qua
/// nhiều lượt thất bại. Guard này xóa best-effort `out_path` khi rời scope (kể cả khi thoát
/// bằng toán tử `?`), TRỪ khi đã `disarm()` trên nhánh thành công — khi đó `out_path` là đầu
/// ra HỢP LỆ và phải được GIỮ lại. Đây là bản đối ứng phía client của bản vá LRC1 ở worker.
struct TempFileGuard {
    path: String,
    armed: bool,
}

impl TempFileGuard {
    fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            armed: true,
        }
    }

    /// Nhả guard: nhánh THÀNH CÔNG gọi hàm này để GIỮ file (không xóa khi drop).
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if self.armed {
            // Best-effort: file có thể chưa kịp được tạo (vd ffmpeg chưa chạy) — bỏ qua lỗi.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// TR-1 (chống path-traversal / đọc file tùy ý): chỉ cho phép thao tác trên file
/// TẠM do CHÍNH app tạo ra — nằm trong thư mục temp của HĐH và mang tiền tố
/// "omnivoice_". Nếu webview bị XSS, nó KHÔNG thể ép Rust đọc file bất kỳ (khóa
/// riêng, tài liệu, mật khẩu…) rồi tuồn ra ngoài.
///
/// `canonicalize` triệt tiêu "..", "." và symlink TRƯỚC khi so khớp — nên không thể
/// lách bằng `temp\..\..\secret`. KHÔNG chặn ký tự ':' (ổ đĩa Windows hợp lệ có dạng
/// "C:\..."); phòng thủ dựa trên tiền tố thư mục đã canonical hóa, không phải lọc ký tự.
fn ensure_owned_temp_file(path: &str) -> Result<PathBuf, String> {
    let canonical = std::fs::canonicalize(path)
        .map_err(|_| "Đường dẫn không hợp lệ hoặc file không tồn tại".to_string())?;
    let temp_root = std::fs::canonicalize(std::env::temp_dir())
        .map_err(|_| "Không xác định được thư mục tạm".to_string())?;
    if !canonical.starts_with(&temp_root) {
        return Err("Từ chối truy cập: chỉ được thao tác trên file tạm của ứng dụng".into());
    }
    let name = canonical.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if !name.starts_with("omnivoice_") {
        return Err("Từ chối truy cập: file không thuộc quyền quản lý của ứng dụng".into());
    }
    Ok(canonical)
}

/// TRẠM CLIENT: tách audio 16kHz mono từ video bằng ffmpeg (sidecar đã bundle).
/// Trả về đường dẫn audio cục bộ + md5 + kích thước để client kiểm tra toàn vẹn
/// trước khi upload AUDIO (không phải video) lên hạ tầng.
#[tauri::command]
fn extract_audio(video_path: String) -> Result<AudioInfo, String> {
    // Đầu vào phải là FILE thật (người dùng chọn qua hộp thoại). is_file() chặn
    // đường dẫn thư mục / chuỗi URL giả trước khi đưa cho ffmpeg.
    if !Path::new(&video_path).is_file() {
        return Err("File video không tồn tại hoặc không phải file hợp lệ".into());
    }
    let out_path = temp_path("audio", "wav");
    // TR-2: dọn out_path trên MỌI đường thoát Err (frontend không nhận path để tự dọn).
    let mut out_guard = TempFileGuard::new(&out_path);

    let output = Command::new_sidecar("ffmpeg")
        .map_err(|e| format!("Không nạp được ffmpeg sidecar: {e}"))?
        .args([
            // TR-1: chỉ cho ffmpeg mở giao thức file/pipe — chặn input playlist/concat
            // độc hại dụ mở http/tcp (SSRF / rò rỉ qua mạng).
            "-protocol_whitelist",
            "file,pipe",
            "-y",
            "-i",
            &video_path,
            "-vn", // bỏ luồng hình ảnh
            "-ar",
            "16000", // 16kHz cho ASR (Whisper)
            "-ac",
            "1", // mono
            &out_path,
        ])
        .output()
        .map_err(|e| format!("Lỗi chạy ffmpeg: {e}"))?;

    if !output.status.success() {
        // Fail-closed: không tạo file giả, báo lỗi thật.
        return Err(format!("ffmpeg thất bại (mã {:?})", output.status.code()));
    }

    let bytes = std::fs::read(&out_path).map_err(|e| format!("Không đọc được audio: {e}"))?;
    if bytes.is_empty() {
        return Err("ffmpeg tạo ra file audio rỗng".into());
    }
    let digest = md5::compute(&bytes);

    // Thành công: out_path là audio hợp lệ — GIỮ lại cho bước upload/mux.
    out_guard.disarm();
    Ok(AudioInfo {
        md5: format!("{:x}", digest),
        size_bytes: bytes.len() as u64,
        audio_path: out_path,
    })
}

/// TRẠM CLIENT: ghép (mux) track audio đã lồng tiếng vào lại VIDEO GỐC cục bộ,
/// giữ nguyên hình ảnh (-c:v copy). Video không rời máy; chỉ audio đã xử lý được
/// tải về rồi ghép tại chỗ.
#[tauri::command]
fn mux_audio_to_video(video_path: String, audio_path: String) -> Result<String, String> {
    // Cả hai đầu vào phải là FILE thật trên đĩa trước khi đưa cho ffmpeg.
    if !Path::new(&video_path).is_file() {
        return Err("File video không tồn tại hoặc không phải file hợp lệ".into());
    }
    if !Path::new(&audio_path).is_file() {
        return Err("File audio không tồn tại hoặc không phải file hợp lệ".into());
    }
    let out_path = temp_path("dubbed", "mp4");
    // TR-2: dọn out_path (mp4 dở dang) nếu mux thoát sớm bằng Err.
    let mut out_guard = TempFileGuard::new(&out_path);

    let output = Command::new_sidecar("ffmpeg")
        .map_err(|e| format!("Không nạp được ffmpeg sidecar: {e}"))?
        .args([
            // TR-1: khóa giao thức về file/pipe cho cả hai input.
            "-protocol_whitelist",
            "file,pipe",
            "-y",
            "-i",
            &video_path,
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            &audio_path,
            "-c:v",
            "copy", // giữ nguyên hình ảnh, không re-encode
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            &out_path,
        ])
        .output()
        .map_err(|e| format!("Lỗi chạy ffmpeg: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "ffmpeg mux thất bại (mã {:?})",
            output.status.code()
        ));
    }

    // Thành công: out_path là video đã ghép — GIỮ để lưu ra Downloads.
    out_guard.disarm();
    Ok(out_path)
}

/// Đọc file AUDIO cục bộ và trả về base64 để webview tải lên lưu trữ đối tượng
/// (R2/S3). Chỉ AUDIO (~30MB) được đọc/tải — KHÔNG bao giờ đọc/tải video gốc.
/// Rust có toàn quyền đọc thư mục tạm (nơi extract_audio ghi ra), tránh giới hạn
/// scope của fs-allowlist phía JS.
#[tauri::command]
fn read_audio_b64(audio_path: String) -> Result<String, String> {
    // TR-1: chỉ đọc file TẠM do app tạo (temp + tiền tố omnivoice_). Nếu webview bị
    // XSS, kẻ tấn công KHÔNG thể gọi read_audio_b64("C:\\...\\secret") để tuồn file.
    let safe = ensure_owned_temp_file(&audio_path)?;
    let bytes = std::fs::read(&safe).map_err(|e| format!("Không đọc được audio: {e}"))?;
    if bytes.is_empty() {
        return Err("File audio rỗng".into());
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// CLIENT-01: ghi AUDIO lồng tiếng tải-về (đưa qua IPC dạng base64) ra file TẠM để
/// bước mux cục bộ dùng. Đặt tiền tố "omnivoice_" nên `cleanup_temp_file` dọn được
/// sau khi ghép xong. Fail-closed: base64 hỏng / audio rỗng => báo lỗi, không tạo
/// file giả.
#[tauri::command]
fn write_temp_audio(audio_b64: String) -> Result<String, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64.as_bytes())
        .map_err(|_| "Dữ liệu audio base64 không hợp lệ".to_string())?;
    if bytes.is_empty() {
        return Err("Audio tải về rỗng".into());
    }
    let out_path = temp_path("dubbed_dl", "wav");
    // TR-2: nếu ghi lỗi giữa chừng (file rỗng/dở đã kịp tạo), dọn best-effort.
    let mut out_guard = TempFileGuard::new(&out_path);
    std::fs::write(&out_path, &bytes).map_err(|e| format!("Không ghi được audio tạm: {e}"))?;
    // Thành công: out_path chứa audio đã tải về — GIỮ cho bước mux cục bộ.
    out_guard.disarm();
    Ok(out_path)
}

/// Các thư mục ĐÍCH được phép chép kết quả ra. Chỉ Downloads — khớp đúng fs-scope mà
/// manifest khai báo (`$DOWNLOAD/*`). Danh sách này là "allowlist thư mục an toàn".
fn allowed_save_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(d) = tauri::api::path::download_dir() {
        roots.push(d);
    }
    roots
}

/// TR-1 (chống GHI file tùy ý / RCE): đích lưu PHẢI nằm trong một thư mục được phép.
/// `dest_path` đến từ hộp thoại Save phía JS, nhưng nếu webview bị XSS kẻ tấn công có
/// thể gọi thẳng IPC `save_output_video` với đích BẤT KỲ (vd thư mục Startup →
/// ghi payload tự khởi động → RCE). Ta canonical hóa THƯ MỤC CHA (đích là file mới nên
/// chính nó chưa tồn tại) để triệt "..": `Downloads\..\..\Startup\x.exe` sau canonical
/// hóa nằm ngoài Downloads => bị từ chối. Trả về đường dẫn đích ĐÃ chuẩn hóa để chép.
fn ensure_dest_in_roots(dest_path: &str, roots: &[PathBuf]) -> Result<PathBuf, String> {
    if dest_path.trim().is_empty() {
        return Err("Chưa chọn nơi lưu".into());
    }
    if roots.is_empty() {
        return Err("Không xác định được thư mục Tải xuống để lưu".into());
    }
    let dest = Path::new(dest_path);
    let file_name = dest
        .file_name()
        .ok_or_else(|| "Đường dẫn lưu thiếu tên file".to_string())?;
    let parent = dest
        .parent()
        .ok_or_else(|| "Đường dẫn lưu không hợp lệ".to_string())?;
    // Thư mục cha PHẢI tồn tại thật để canonical hóa (triệt symlink/".."/"." trước khi khớp).
    let parent_canon =
        std::fs::canonicalize(parent).map_err(|_| "Thư mục lưu không tồn tại".to_string())?;
    for root in roots {
        if let Ok(root_canon) = std::fs::canonicalize(root) {
            if parent_canon.starts_with(&root_canon) {
                return Ok(parent_canon.join(file_name));
            }
        }
    }
    Err("Từ chối: chỉ được lưu video vào thư mục Tải xuống (Downloads)".into())
}

/// CLIENT-01: chép video KẾT QUẢ (đang nằm trong temp, do mux tạo) ra vị trí NGƯỜI
/// DÙNG chọn qua hộp thoại Save. NGUỒN phải là file tạm của app (guard chống lệnh
/// webview đọc file tùy ý). ĐÍCH bị giới hạn trong thư mục Downloads — KHÔNG tin cậy
/// `dest_path` một cách mù quáng, vì webview bị XSS có thể gọi IPC với đích bất kỳ.
#[tauri::command]
fn save_output_video(temp_path: String, dest_path: String) -> Result<(), String> {
    let src = ensure_owned_temp_file(&temp_path)?;
    let safe_dest = ensure_dest_in_roots(&dest_path, &allowed_save_roots())?;
    std::fs::copy(&src, &safe_dest).map_err(|e| format!("Không lưu được video: {e}"))?;
    Ok(())
}

/// TR-2 (dọn rác temp): xóa file TẠM do app tạo sau khi đã dùng xong (giải phóng
/// ~30MB audio/temp mỗi lần xử lý, tránh phình thư mục tạm qua nhiều lượt). Best-effort:
/// nếu file đã biến mất thì coi như đã sạch (Ok), CHỈ báo lỗi khi xóa thật bại.
///
/// Vẫn đi qua `ensure_owned_temp_file`: nếu webview bị XSS, kẻ tấn công KHÔNG thể gọi
/// cleanup_temp_file("C:\\...\\quan-trọng") để xóa file tùy ý của người dùng — chỉ file
/// nằm trong thư mục temp và mang tiền tố "omnivoice_" mới bị đụng tới.
#[tauri::command]
fn cleanup_temp_file(path: String) -> Result<(), String> {
    let safe = match ensure_owned_temp_file(&path) {
        Ok(p) => p,
        // canonicalize thất bại (file đã bị xóa / đường dẫn không tồn tại) => đã sạch.
        // Guard vẫn giữ nguyên: ta CHƯA xóa gì cả nên không có rủi ro an toàn.
        Err(_) => return Ok(()),
    };
    match std::fs::remove_file(&safe) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("Không xóa được file tạm: {e}")),
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            extract_audio,
            mux_audio_to_video,
            read_audio_b64,
            write_temp_audio,
            save_output_video,
            cleanup_temp_file
        ])
        .run(tauri::generate_context!())
        .expect("Lỗi khởi chạy OmniVoice Tauri Application");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_suffix() -> u128 {
        // std::process + thời gian: đủ để tránh đụng tên giữa các lần chạy test.
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            ^ (std::process::id() as u128)
    }

    fn make_temp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let mut f = std::fs::File::create(&p).expect("tạo file tạm test");
        f.write_all(b"x").unwrap();
        p
    }

    #[test]
    fn accepts_owned_temp_file_with_prefix() {
        // File THẬT trong temp + tiền tố "omnivoice_" => được chấp nhận.
        let name = format!("omnivoice_unit_{}.wav", unique_suffix());
        let p = make_temp(&name);
        let out = ensure_owned_temp_file(p.to_str().unwrap());
        let _ = std::fs::remove_file(&p);
        assert!(out.is_ok(), "file temp hợp lệ phải qua guard: {out:?}");
    }

    #[test]
    fn rejects_temp_file_without_prefix() {
        // Trong temp NHƯNG sai tiền tố => từ chối (không phải file do app tạo).
        let name = format!("notours_unit_{}.wav", unique_suffix());
        let p = make_temp(&name);
        let out = ensure_owned_temp_file(p.to_str().unwrap());
        let _ = std::fs::remove_file(&p);
        assert!(out.is_err(), "sai tiền tố phải bị từ chối");
    }

    #[test]
    fn rejects_file_outside_temp_root() {
        // File THẬT nhưng NẰM NGOÀI thư mục temp (Cargo.toml của chính crate) => từ chối.
        // Đây là lõi chống path-traversal: dù có tồn tại thật cũng không được đọc/xóa.
        let outside = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let out = ensure_owned_temp_file(outside);
        assert!(out.is_err(), "file ngoài temp phải bị từ chối");
    }

    #[test]
    fn rejects_nonexistent_path() {
        // canonicalize thất bại cho đường dẫn không tồn tại => từ chối (không suy đoán).
        let out = ensure_owned_temp_file("Z:\\khong-ton-tai\\omnivoice_ghost.wav");
        assert!(out.is_err(), "đường dẫn không tồn tại phải bị từ chối");
    }

    #[test]
    fn cleanup_is_idempotent_on_missing_file() {
        // TR-2 best-effort: xóa file không tồn tại (đường dẫn temp hợp lệ nhưng đã mất)
        // KHÔNG được coi là lỗi.
        let name = format!("omnivoice_gone_{}.wav", unique_suffix());
        let mut p = std::env::temp_dir();
        p.push(&name); // không tạo file
        let out = cleanup_temp_file(p.to_string_lossy().into_owned());
        assert!(
            out.is_ok(),
            "cleanup file đã mất phải Ok (best-effort): {out:?}"
        );
    }

    #[test]
    fn cleanup_removes_owned_temp_file() {
        // TR-2: file temp hợp lệ do app tạo => cleanup xóa thật và trả Ok.
        let name = format!("omnivoice_del_{}.wav", unique_suffix());
        let p = make_temp(&name);
        assert!(p.exists());
        let out = cleanup_temp_file(p.to_string_lossy().into_owned());
        assert!(out.is_ok(), "cleanup phải Ok: {out:?}");
        assert!(!p.exists(), "file tạm phải bị xóa sau cleanup");
    }

    #[test]
    fn cleanup_refuses_file_outside_temp() {
        // TR-2 an toàn: cleanup KHÔNG được xóa file ngoài temp. Cargo.toml phải còn nguyên.
        let outside = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let _ = cleanup_temp_file(outside.to_string());
        assert!(
            Path::new(outside).is_file(),
            "file ngoài temp KHÔNG được bị xóa"
        );
    }

    #[test]
    fn temp_guard_removes_file_on_drop_when_armed() {
        // TR-2: guard còn "armed" (nhánh Err, chưa disarm) => xóa out_path khi rời scope.
        let name = format!("omnivoice_guard_armed_{}.wav", unique_suffix());
        let p = make_temp(&name);
        assert!(p.exists());
        {
            let _g = TempFileGuard::new(p.to_str().unwrap());
            // rời scope mà KHÔNG disarm (mô phỏng đường thoát Err) -> phải xóa.
        }
        assert!(!p.exists(), "guard còn armed phải xóa file khi drop");
    }

    #[test]
    fn temp_guard_keeps_file_after_disarm() {
        // Nhánh THÀNH CÔNG gọi disarm() => out_path (đầu ra hợp lệ) phải được GIỮ.
        let name = format!("omnivoice_guard_disarm_{}.wav", unique_suffix());
        let p = make_temp(&name);
        assert!(p.exists());
        {
            let mut g = TempFileGuard::new(p.to_str().unwrap());
            g.disarm();
        }
        assert!(p.exists(), "sau disarm phải GIỮ file (đầu ra hợp lệ)");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn temp_guard_drop_missing_file_is_noop() {
        // Best-effort: file chưa từng được tạo (ffmpeg chưa chạy) => drop không panic.
        let name = format!("omnivoice_guard_ghost_{}.wav", unique_suffix());
        let mut p = std::env::temp_dir();
        p.push(&name); // KHÔNG tạo file
        {
            let _g = TempFileGuard::new(p.to_str().unwrap());
        }
        assert!(!p.exists(), "file ma không được vô tình tạo ra");
    }

    #[test]
    fn write_temp_audio_decodes_base64_to_owned_temp() {
        // CLIENT-01: base64 hợp lệ => ghi ĐÚNG bytes ra file temp có tiền tố omnivoice_,
        // và file đó qua được guard (để cleanup dọn về sau).
        let raw = b"RIFF....WAVEfake-bytes";
        let b64 = base64::engine::general_purpose::STANDARD.encode(raw);
        let path = write_temp_audio(b64).expect("phải ghi được");
        assert!(
            ensure_owned_temp_file(&path).is_ok(),
            "file ghi ra phải qua guard"
        );
        let back = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&back, raw, "bytes phải khớp hệt sau round-trip base64");
    }

    #[test]
    fn write_temp_audio_rejects_bad_base64() {
        // Fail-closed: base64 rác => lỗi, KHÔNG tạo file giả.
        let out = write_temp_audio("@@@not-base64@@@".into());
        assert!(out.is_err(), "base64 hỏng phải bị từ chối");
    }

    #[test]
    fn save_output_video_copies_into_downloads() {
        // CLIENT-01: nguồn là file temp của app + đích nằm trong Downloads => chép thành công.
        let dl = match allowed_save_roots().into_iter().find(|d| d.exists()) {
            Some(d) => d,
            // Không có thư mục Downloads (môi trường tối giản) => BỎ QUA, không fail giả.
            None => return,
        };
        let name = format!("omnivoice_out_{}.mp4", unique_suffix());
        let src = make_temp(&name);
        let dest = dl.join(format!("omnivoice_test_save_{}.mp4", unique_suffix()));
        let out = save_output_video(
            src.to_string_lossy().into_owned(),
            dest.to_string_lossy().into_owned(),
        );
        let ok = out.is_ok() && dest.exists();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dest);
        assert!(ok, "phải chép được file kết quả vào Downloads: {out:?}");
    }

    #[test]
    fn save_output_video_refuses_non_owned_source() {
        // An toàn: KHÔNG được chép (tuồn) file ngoài temp ra ngoài theo lệnh webview.
        // Guard nguồn chạy TRƯỚC guard đích nên đây kiểm đúng nhánh "nguồn không hợp lệ".
        let outside = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let mut dest = std::env::temp_dir();
        dest.push(format!("exfil_{}.txt", unique_suffix()));
        let out = save_output_video(outside.to_string(), dest.to_string_lossy().into_owned());
        let leaked = dest.exists();
        let _ = std::fs::remove_file(&dest);
        assert!(out.is_err(), "nguồn ngoài temp phải bị từ chối");
        assert!(!leaked, "KHÔNG được tạo bản sao file ngoài temp");
    }

    #[test]
    fn ensure_dest_in_roots_accepts_dest_in_allowed_root() {
        // Đích có thư mục cha = root cho phép (dùng temp làm root để không lệ thuộc Downloads).
        let root = std::env::temp_dir();
        let dest = root.join(format!("omnivoice_dest_{}.mp4", unique_suffix()));
        let out = ensure_dest_in_roots(&dest.to_string_lossy(), std::slice::from_ref(&root));
        assert!(
            out.is_ok(),
            "đích trong root cho phép phải được chấp nhận: {out:?}"
        );
    }

    #[test]
    fn ensure_dest_in_roots_rejects_dest_outside_roots() {
        // Đích nằm trong temp nhưng root cho phép là thư mục KHÁC => từ chối (không ghi ra ngoài).
        let dest = std::env::temp_dir().join(format!("omnivoice_dest_{}.mp4", unique_suffix()));
        let other_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let out = ensure_dest_in_roots(&dest.to_string_lossy(), &[other_root]);
        assert!(out.is_err(), "đích ngoài mọi root phải bị từ chối");
    }

    #[test]
    fn ensure_dest_in_roots_rejects_traversal_escape() {
        // "<temp>/../evil.exe": cha canonical hóa thành cha-của-temp (ngoài root) => từ chối.
        // Đây là lõi chống ghi-file-tùy-ý: không thể lách allowlist bằng "..".
        let root = std::env::temp_dir();
        let mut traversal = root.clone();
        traversal.push("..");
        traversal.push(format!("omnivoice_evil_{}.exe", unique_suffix()));
        let out = ensure_dest_in_roots(&traversal.to_string_lossy(), std::slice::from_ref(&root));
        assert!(
            out.is_err(),
            "đường dẫn traversal thoát khỏi root phải bị từ chối"
        );
    }

    #[test]
    fn ensure_dest_in_roots_rejects_empty() {
        let root = std::env::temp_dir();
        assert!(
            ensure_dest_in_roots("   ", &[root]).is_err(),
            "đích rỗng phải bị từ chối"
        );
    }
}
