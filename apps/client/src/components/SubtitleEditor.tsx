
import { FileEdit, MessageSquare, Clock } from 'lucide-react';

export interface Subtitle {
  id: string;
  speaker: string;
  start: string;
  end: string;
  text: string;
}

interface SubtitleEditorProps {
  subtitles: Subtitle[];
  onChange: (id: string, newText: string) => void;
  onSpeakerChange: (id: string, newSpeaker: string) => void;
}

export function SubtitleEditor({ subtitles, onChange, onSpeakerChange }: SubtitleEditorProps) {
  return (
    <div className="bg-card text-card-foreground border rounded-2xl p-6 shadow-sm w-full">
      <div className="flex items-center gap-3 mb-6 border-b pb-4">
        <FileEdit className="w-6 h-6 text-primary" />
        <div>
          <h2 className="text-xl font-bold">Trình Soạn Thảo Phụ Đề</h2>
          <p className="text-sm text-muted-foreground">Kiểm tra và sửa lỗi dịch thuật của AI trước khi lồng tiếng.</p>
        </div>
      </div>
      
      <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
        {subtitles.map((sub) => (
          <div key={sub.id} className="flex gap-4 p-4 rounded-xl bg-background border hover:border-primary/50 transition-colors">
            <div className="flex flex-col items-center justify-center gap-2 min-w-[96px]">
              <input
                type="text"
                value={sub.speaker}
                onChange={(e) => onSpeakerChange(sub.id, e.target.value)}
                aria-label="Người nói"
                title="Sửa tên người nói — mỗi tên khác nhau sẽ được gán một giọng riêng ở bước dưới."
                className="w-full px-2 py-1 bg-muted rounded-lg text-xs font-bold text-center text-muted-foreground border border-transparent focus:border-ring focus:bg-background outline-none transition-colors"
              />
              <div className="flex items-center gap-1 text-[10px] text-muted-foreground font-mono">
                <Clock className="w-3 h-3" />
                {sub.start} - {sub.end}
              </div>
            </div>
            
            <div className="flex-1">
              <div className="relative">
                <MessageSquare className="absolute top-3 left-3 w-4 h-4 text-muted-foreground" />
                <textarea
                  value={sub.text}
                  onChange={(e) => onChange(sub.id, e.target.value)}
                  className="w-full bg-background border border-input rounded-lg p-3 pl-10 text-sm focus:ring-2 focus:ring-ring focus:border-ring outline-none resize-none transition-all h-20"
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
