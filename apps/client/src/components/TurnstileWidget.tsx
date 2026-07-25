import { useEffect, useRef } from 'react';

/**
 * Widget Cloudflare Turnstile (chống bot ở bước đăng ký thiết bị — Trạm 4).
 *
 * Chỉ hiển thị khi có `siteKey` (deployer đặt VITE_TURNSTILE_SITE_KEY). Ở môi trường
 * dev không cấu hình, widget không render và Gateway bỏ qua Turnstile (fail-open dev,
 * fail-closed prod — khớp với verifyTurnstile phía Gateway).
 */

declare global {
  interface Window {
    turnstile?: {
      render: (
        el: HTMLElement,
        opts: { sitekey: string; callback: (token: string) => void; 'error-callback'?: () => void },
      ) => string;
      remove: (id: string) => void;
    };
  }
}

const SCRIPT_SRC = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';

function ensureScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (window.turnstile) return resolve();
    const existing = document.querySelector(`script[src="${SCRIPT_SRC}"]`);
    if (existing) {
      existing.addEventListener('load', () => resolve());
      existing.addEventListener('error', () => reject(new Error('Không tải được Turnstile')));
      return;
    }
    const s = document.createElement('script');
    s.src = SCRIPT_SRC;
    s.async = true;
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error('Không tải được Turnstile'));
    document.head.appendChild(s);
  });
}

export function TurnstileWidget({
  siteKey,
  onToken,
}: {
  siteKey: string;
  onToken: (token: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetIdRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    ensureScript()
      .then(() => {
        if (cancelled || !containerRef.current || !window.turnstile) return;
        widgetIdRef.current = window.turnstile.render(containerRef.current, {
          sitekey: siteKey,
          callback: (token) => onToken(token),
          'error-callback': () => onToken(''),
        });
      })
      .catch(() => {
        // Không tải được script -> không cấp token (fail-closed): đăng ký sẽ bị chặn.
      });
    return () => {
      cancelled = true;
      if (widgetIdRef.current && window.turnstile) {
        window.turnstile.remove(widgetIdRef.current);
      }
    };
  }, [siteKey, onToken]);

  return <div ref={containerRef} className="my-2" />;
}
