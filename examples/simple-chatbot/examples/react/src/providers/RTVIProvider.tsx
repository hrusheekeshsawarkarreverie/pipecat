import { type PropsWithChildren } from 'react';
import { RTVIClient } from 'realtime-ai';
import { DailyTransport } from '@daily-co/realtime-ai-daily';
import { RTVIClientProvider } from 'realtime-ai-react';

const transport = new DailyTransport();

const client = new RTVIClient({
  transport,
  params: {
    baseUrl: 'http://172.18.0.04:85',
    endpoints: {
      // connect: '/connect',
      connect: '/webrtc/connect/125881',
    },
  },
  enableMic: true,
  enableCam: false,
});

export function RTVIProvider({ children }: PropsWithChildren) {
  return <RTVIClientProvider client={client}>{children}</RTVIClientProvider>;
}
