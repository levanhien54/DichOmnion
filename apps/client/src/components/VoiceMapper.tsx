
import { Mic, UserCog, Volume2 } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

interface VoiceMapperProps {
  speakers: string[];
  mapping: Record<string, string>;
  onChange: (speaker: string, voiceId: string) => void;
}

const AVAILABLE_VOICES = [
  { id: 'nam_tram', name: 'Nam Trầm (Điện ảnh)' },
  { id: 'nam_tre', name: 'Nam Trẻ (Năng động)' },
  { id: 'nu_cao', name: 'Nữ Cao (Nhẹ nhàng)' },
  { id: 'nu_truyen_cam', name: 'Nữ (Truyền cảm)' },
  { id: 'tre_em', name: 'Trẻ em (Dễ thương)' },
];

export function VoiceMapper({ speakers, mapping, onChange }: VoiceMapperProps) {
  return (
    <div className="bg-card text-card-foreground border rounded-2xl p-6 shadow-sm w-full">
      <div className="flex items-center gap-3 mb-6 border-b pb-4">
        <UserCog className="w-6 h-6 text-primary" />
        <div>
          <h2 className="text-xl font-bold">Gán Giọng Nhân Vật</h2>
          <p className="text-sm text-muted-foreground">Chọn giọng nói TTS phù hợp cho từng người nói trong video.</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {speakers.map((speaker) => (
          <div key={speaker} className="flex items-center justify-between p-4 bg-background border rounded-xl hover:border-primary/50 transition-colors">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center border">
                <Mic className="w-5 h-5 text-muted-foreground" />
              </div>
              <span className="font-bold">{speaker}</span>
            </div>
            
            <div className="w-[180px]">
              <Select
                value={mapping[speaker] || ""}
                onValueChange={(val) => onChange(speaker, val)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="-- Chọn Giọng --" />
                </SelectTrigger>
                <SelectContent>
                  {AVAILABLE_VOICES.map((v) => (
                    <SelectItem key={v.id} value={v.id}>
                      <div className="flex items-center gap-2">
                        <Volume2 className="w-4 h-4 text-muted-foreground" />
                        {v.name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
