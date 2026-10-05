// No-op action logger for the standalone demo.
// The real play-zendo app records user actions here for the human study;
// the demo just discards them.
export function log(_action: string, _payload?: Record<string, unknown>): void {}
