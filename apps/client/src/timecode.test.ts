import { describe, it, expect } from 'vitest';
import { timecodeToSeconds } from './timecode';

// Hàm thuần, không DOM — chạy trong môi trường 'node' mặc định của vitest.
// Bảo vệ hợp đồng CC-1: mọi hình dạng người dùng nhập PHẢI ra GIÂY (số), phản
// chiếu 1-1 src/timecode.py phía worker.
describe('timecodeToSeconds', () => {
  it('trả nguyên số giây khi nhập đã là số', () => {
    expect(timecodeToSeconds('83')).toBe(83);
    expect(timecodeToSeconds('0')).toBe(0);
    expect(timecodeToSeconds('3600')).toBe(3600);
  });

  it('chấp nhận thập phân với cả dấu chấm lẫn dấu phẩy', () => {
    expect(timecodeToSeconds('1.5')).toBe(1.5);
    expect(timecodeToSeconds('1,5')).toBe(1.5);
  });

  it('quy MM:SS về giây', () => {
    expect(timecodeToSeconds('01:30')).toBe(90);
    expect(timecodeToSeconds('00:05')).toBe(5);
    expect(timecodeToSeconds('10:00')).toBe(600);
  });

  it('quy HH:MM:SS về giây', () => {
    expect(timecodeToSeconds('00:00:05')).toBe(5);
    expect(timecodeToSeconds('00:01:30')).toBe(90);
    expect(timecodeToSeconds('01:00:00')).toBe(3600);
    expect(timecodeToSeconds('01:02:03')).toBe(3723);
  });

  it('cắt khoảng trắng thừa trước khi phân tích', () => {
    expect(timecodeToSeconds('  00:00:05  ')).toBe(5);
    expect(timecodeToSeconds('  42 ')).toBe(42);
  });

  it('trả 0 cho chuỗi rỗng / chỉ khoảng trắng / null-ish (không đoán bừa)', () => {
    expect(timecodeToSeconds('')).toBe(0);
    expect(timecodeToSeconds('   ')).toBe(0);
    // @ts-expect-error kiểm tra bảo vệ runtime khi tc == null
    expect(timecodeToSeconds(null)).toBe(0);
    // @ts-expect-error kiểm tra bảo vệ runtime khi tc == undefined
    expect(timecodeToSeconds(undefined)).toBe(0);
  });

  it('trả 0 khi có thành phần không phải số (rác -> 0, KHÔNG rơi về pacing sai)', () => {
    expect(timecodeToSeconds('garbage')).toBe(0);
    expect(timecodeToSeconds('1:2:x')).toBe(0);
    expect(timecodeToSeconds('aa:bb')).toBe(0);
  });
});
