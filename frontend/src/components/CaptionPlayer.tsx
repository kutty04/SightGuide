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
      <div className="flex items-center gap-2 text-xs font-mono text-amber-400 bg-amber-950/40 px-3 py-2 border border-amber-800/50">
        <VolumeX className="w-4 h-4 flex-shrink-0" />
        <span>Speech synthesis is not supported by your browser environment.</span>
      </div>
    );
  }

  return (
    <section
      aria-label={label}
      className="bg-[#0c0c0c] border border-white/10 p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-md"
    >
      <div className="flex items-center gap-3">
        <div
          className={`w-7 h-7 border flex items-center justify-center transition-colors ${
            isPlaying ? 'bg-[#D4FF00] text-black border-[#D4FF00] animate-pulse' : 'bg-black text-[#D4FF00] border-white/20'
          }`}
          aria-hidden="true"
        >
          <Volume2 className="w-3.5 h-3.5" />
        </div>
        <div>
          <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-400 block">
            {label}
          </span>
          <span className="text-xs font-mono text-white">
            {isPlaying ? 'PLAYING AUDITORY NARRATIVE...' : isPaused ? 'AUDIO PAUSED' : 'READY TO SYNTHESIZE'}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {!isPlaying ? (
          <button
            type="button"
            onClick={handlePlay}
            disabled={!captionText}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#D4FF00] hover:bg-[#ccff00] disabled:opacity-40 disabled:pointer-events-none text-black text-xs font-mono font-bold tracking-wider uppercase transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
            aria-label="Play caption aloud"
          >
            <Play className="w-3 h-3 fill-current" />
            <span>{isPaused ? 'RESUME' : 'SPEAK'}</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={handlePause}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-400 hover:bg-amber-300 text-black text-xs font-mono font-bold tracking-wider uppercase transition focus-visible:outline-2 focus-visible:outline-amber-400"
            aria-label="Pause speech"
          >
            <Pause className="w-3 h-3 fill-current" />
            <span>PAUSE</span>
          </button>
        )}

        <button
          type="button"
          onClick={handleStop}
          disabled={!isPlaying && !isPaused}
          className="p-1.5 bg-black hover:bg-white/10 text-zinc-300 disabled:opacity-30 disabled:pointer-events-none border border-white/20 hover:border-[#D4FF00] transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
          aria-label="Stop audio playback"
          title="Stop audio"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        <div className="flex items-center gap-1.5 ml-2 border-l border-white/10 pl-3">
          <label htmlFor="rate-select" className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            Speed:
          </label>
          <select
            id="rate-select"
            value={rate}
            onChange={(e) => setRate(parseFloat(e.target.value))}
            className="bg-black border border-white/20 text-white text-xs font-mono px-2 py-1 focus:ring-1 focus:ring-[#D4FF00] focus:border-[#D4FF00]"
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

export default CaptionPlayer;
