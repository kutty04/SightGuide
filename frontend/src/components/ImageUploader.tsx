import React, { useEffect, useRef, useState } from 'react';
import { Upload, Image as ImageIcon, Sliders, AlertCircle, FileCheck } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File, lambda: number) => void;
  isLoading: boolean;
  lambdaWeight: number;
  onLambdaChange: (val: number) => void;
}

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_BYTES = 10 * 1024 * 1024; // 10MB

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelected,
  isLoading,
  lambdaWeight,
  onLambdaChange,
}) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isScanning) {
      const timer = setTimeout(() => {
        setIsScanning(false);
      }, 1400);
      return () => clearTimeout(timer);
    }
  }, [isScanning]);

  const validateAndSetFile = (file: File) => {
    setErrorMsg(null);
    if (!ALLOWED_TYPES.includes(file.type)) {
      setErrorMsg('Unsupported format. Please upload a JPEG, PNG, or WebP image.');
      return;
    }
    if (file.size > MAX_BYTES) {
      setErrorMsg('File too large. Maximum supported size is 10 MB.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const prefersReducedMotion = typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!prefersReducedMotion) {
      setIsScanning(true);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFile) {
      onImageSelected(selectedFile, lambdaWeight);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Drag & Drop Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed p-6 text-center cursor-pointer transition-all ${
          dragActive
            ? 'border-[#D4FF00] bg-[#D4FF00]/5'
            : 'border-white/20 hover:border-[#D4FF00]/60 bg-[#0c0c0c]'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleChange}
          className="hidden"
          aria-label="Upload an image for captioning"
        />

        {previewUrl ? (
          <div className="space-y-3">
            <div className="relative inline-block max-h-64 overflow-hidden border border-white/20 bg-black reticle-bracket">
              {/* Laser Scanline on first preview reveal */}
              {isScanning && (
                <>
                  <div
                    className="animate-laser-sweep pointer-events-none absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-[#D4FF00] to-transparent shadow-[0_0_12px_#D4FF00] z-20"
                    aria-hidden="true"
                  />
                  <div
                    className="pointer-events-none absolute top-2 right-2 px-2 py-0.5 bg-black/80 border border-[#D4FF00]/60 text-[10px] font-mono text-[#D4FF00] z-20 tracking-wider"
                    aria-hidden="true"
                  >
                    OPTICAL SCAN
                  </div>
                </>
              )}

              {/* Active backend processing overlay */}
              {isLoading && (
                <div
                  className="pointer-events-none absolute inset-0 bg-black/70 backdrop-blur-[2px] flex items-center justify-center z-20"
                  aria-hidden="true"
                >
                  <div className="flex items-center gap-2.5 px-3.5 py-2 bg-black border border-[#D4FF00] text-[#D4FF00] text-xs font-mono">
                    <div className="w-3.5 h-3.5 border-2 border-[#D4FF00]/30 border-t-[#D4FF00] rounded-full animate-spin" />
                    <span className="tracking-widest">EXTRACTING EMBEDDINGS</span>
                  </div>
                </div>
              )}

              <img
                src={previewUrl}
                alt="Selected image preview"
                className="max-h-64 w-auto object-contain mx-auto block"
              />
            </div>
            <div className="flex items-center justify-center gap-2 text-xs text-zinc-300">
              <FileCheck className="w-4 h-4 text-[#D4FF00]" />
              <span className="font-mono text-white">{selectedFile?.name}</span>
              <span className="text-zinc-500 font-mono">
                ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
              </span>
            </div>
            <p className="text-[11px] text-zinc-500 font-mono uppercase tracking-wider">
              Click or drop another image to replace
            </p>
          </div>
        ) : (
          <div className="space-y-3 py-6">
            <div className="w-12 h-12 bg-black border border-white/20 text-[#D4FF00] flex items-center justify-center mx-auto">
              <Upload className="w-5 h-5" />
            </div>
            <div className="text-sm font-bold text-white tracking-wide font-sans">
              Drop target image here, or <span className="text-[#D4FF00] underline underline-offset-4">browse files</span>
            </div>
            <p className="text-[11px] text-zinc-500 font-mono tracking-wider uppercase">
              Supports JPEG, PNG, or WebP up to 10 MB
            </p>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="flex items-center gap-2.5 p-3 bg-red-950/60 border border-red-700 text-red-200 text-xs font-mono">
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-red-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Diversity Lambda Parameter Control */}
      <div className="bg-[#0c0c0c] border border-white/10 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <label htmlFor="lambda-slider" className="flex items-center gap-2 text-xs font-mono font-bold text-zinc-300 uppercase tracking-wider">
            <Sliders className="w-3.5 h-3.5 text-[#D4FF00]" />
            <span>Diversity Trade-off Weight (&lambda; = {lambdaWeight.toFixed(2)})</span>
          </label>
          <span className="text-xs text-[#D4FF00] font-mono font-bold">
            {lambdaWeight === 1.0
              ? '[CONTROL: NAIVE TOP-2]'
              : lambdaWeight === 0.5
              ? '[BALANCED: RECOMMENDED]'
              : lambdaWeight < 0.5
              ? '[MAX DIVERSITY]'
              : '[MAX GROUNDING]'}
          </span>
        </div>

        <input
          id="lambda-slider"
          type="range"
          min="0.0"
          max="1.0"
          step="0.05"
          value={lambdaWeight}
          onChange={(e) => onLambdaChange(parseFloat(e.target.value))}
          className="w-full accent-[#D4FF00] cursor-pointer"
        />

        <div className="flex justify-between text-[10px] text-zinc-500 font-mono">
          <span>&lambda;=0.0 (Max Novelty)</span>
          <span className="text-[#D4FF00]">&lambda;=0.5 (Primary Balanced)</span>
          <span>&lambda;=1.0 (Naive CLIP Top-2)</span>
        </div>
      </div>

      {/* Action Button */}
      <button
        type="submit"
        disabled={!selectedFile || isLoading}
        className="w-full py-3.5 px-4 bg-[#D4FF00] hover:bg-[#ccff00] disabled:opacity-40 disabled:pointer-events-none text-black text-xs font-mono font-bold tracking-widest uppercase flex items-center justify-center gap-2 transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
      >
        {isLoading ? (
          <>
            <div className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
            <span>EXTRACTING CANDIDATES &amp; EVALUATING METHODS...</span>
          </>
        ) : (
          <>
            <ImageIcon className="w-4 h-4 text-black" />
            <span>GENERATE &amp; COMPARE CAPTIONS</span>
          </>
        )}
      </button>
    </form>
  );
};

export default ImageUploader;
