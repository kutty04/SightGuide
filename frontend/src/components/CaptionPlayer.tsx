import React, { useEffect, useState } from 'react';
import { Volume2, VolumeX, Pause, Play, RotateCcw } from 'lucide-react';

interface CaptionPlayerProps {
  captionText: string;
  label?: string;
}

export const CaptionPlayer: React.FC<CaptionPlayerProps> = ({
  captionText,
  label = 'Accessibility Audio Playback',
}) => {
  const [isSupported, setIsSupported] = useState<boolean>(true);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [rate, setRate] = useState<number>(1.0);

  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      setIsSupported(false);
    }

    return () => {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const handlePlay = () => {
    if (!isSupported || !captionText) return;

    if (isPaused) {
      window.speechSynthesis.resume();
      setIsPaused(false);
      setIsPlaying(true);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(captionText);
    utterance.rate = rate;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      setIsPlaying(true);
      setIsPaused(false);
    };

    utterance.onend = () => {
      setIsPlaying(false);
      setIsPaused(false);
    };

    utterance.onerror = () => {
      setIsPlaying(false);
      setIsPaused(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  const handlePause = () => {
    if (!isSupported || !isPlaying) return;
    window.speechSynthesis.pause();
    setIsPaused(true);
    setIsPlaying(false);
  };

  const handleStop = () => {
    if (!isSupported) return;
    window.speechSynthesis.cancel();
    setIsPlaying(false);
    setIsPaused(false);
  };

  if (!isSupported) {
    return (
      <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-950/40 px-3 py-2 rounded-lg border border-amber-800/50">
        <VolumeX className="w-4 h-4" />
        <span>Speech synthesis is not supported by your browser.</span>
      </div>
    );
  }

  return (
    <section
      aria-label={label}
      className="bg-slate-900/80 border border-slate-700/80 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-sm"
    >
      <div className="flex items-center gap-2.5">
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors ${
            isPlaying ? 'bg-indigo-500 text-white animate-pulse' : 'bg-slate-800 text-slate-300'
          }`}
          aria-hidden="true"
        >
          <Volume2 className="w-4 h-4" />
        </div>
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
            {label}
          </span>
          <span className="text-xs text-slate-400">
            {isPlaying ? 'Playing caption audio...' : isPaused ? 'Audio paused' : 'Ready to speak'}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {!isPlaying ? (
          <button
            type="button"
            onClick={handlePlay}
            disabled={!captionText}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:pointer-events-none text-white text-xs font-medium rounded-lg shadow-sm transition focus-visible:outline-2 focus-visible:outline-indigo-400"
            aria-label="Play caption aloud"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isPaused ? 'Resume' : 'Speak'}</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={handlePause}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-xs font-medium rounded-lg shadow-sm transition focus-visible:outline-2 focus-visible:outline-amber-400"
            aria-label="Pause speech"
          >
            <Pause className="w-3.5 h-3.5 fill-current" />
            <span>Pause</span>
          </button>
        )}

        <button
          type="button"
          onClick={handleStop}
          disabled={!isPlaying && !isPaused}
          className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40 disabled:pointer-events-none rounded-lg border border-slate-700 transition focus-visible:outline-2 focus-visible:outline-indigo-400"
          aria-label="Stop audio playback"
          title="Stop audio"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        <div className="flex items-center gap-1.5 ml-2 border-l border-slate-700 pl-3">
          <label htmlFor="rate-select" className="text-xs text-slate-400">
            Speed:
          </label>
          <select
            id="rate-select"
            value={rate}
            onChange={(e) => setRate(parseFloat(e.target.value))}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded px-1.5 py-1 focus:ring-1 focus:ring-indigo-500"
            aria-label="Playback speed"
          >
            <option value="0.8">0.8x</option>
            <option value="1.0">1.0x</option>
            <option value="1.2">1.2x</option>
            <option value="1.5">1.5x</option>
          </select>
        </div>
      </div>
    </section>
  );
};
