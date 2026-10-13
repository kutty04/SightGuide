import React, { useRef, useState } from 'react';
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
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
          dragActive
            ? 'border-indigo-500 bg-indigo-950/20'
            : 'border-slate-700 hover:border-slate-600 bg-slate-900/60'
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
            <div className="relative inline-block max-h-64 rounded-xl overflow-hidden shadow-md border border-slate-700 bg-black">
              <img
                src={previewUrl}
                alt="Selected image preview"
                className="max-h-64 w-auto object-contain mx-auto"
              />
            </div>
            <div className="flex items-center justify-center gap-2 text-xs text-slate-300">
              <FileCheck className="w-4 h-4 text-emerald-400" />
              <span className="font-medium text-slate-200">{selectedFile?.name}</span>
              <span className="text-slate-400">
                ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
              </span>
            </div>
            <p className="text-xs text-slate-400">Click or drop another image to replace</p>
          </div>
        ) : (
          <div className="space-y-2 py-4">
            <div className="w-12 h-12 bg-indigo-950/60 border border-indigo-700/50 text-indigo-400 rounded-full flex items-center justify-center mx-auto">
              <Upload className="w-5 h-5" />
            </div>
            <div className="text-sm font-semibold text-slate-200">
              Drop your image here, or <span className="text-indigo-400 underline">browse</span>
            </div>
            <p className="text-xs text-slate-400">
              Supports JPEG, PNG, or WebP up to 10 MB
            </p>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="flex items-center gap-2 p-3 bg-red-950/50 border border-red-800 text-red-200 text-xs rounded-lg">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Diversity Lambda Parameter Control */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <label htmlFor="lambda-slider" className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
            <Sliders className="w-4 h-4 text-indigo-400" />
            Diversity Trade-off Weight (&lambda; = {lambdaWeight.toFixed(2)})
          </label>
          <span className="text-xs text-indigo-300 font-mono font-medium">
            {lambdaWeight === 1.0
              ? 'Control (Naive Top-2)'
              : lambdaWeight === 0.5
              ? 'Primary Balanced (Recommended)'
              : lambdaWeight < 0.5
              ? 'High Diversity Emphasis'
              : 'High Grounding Emphasis'}
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
          className="w-full accent-indigo-500 cursor-pointer"
        />

        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>&lambda;=0.0 (Max Diversity)</span>
          <span>&lambda;=0.5 (Primary Balanced)</span>
          <span>&lambda;=1.0 (Naive CLIP Top-2)</span>
        </div>
      </div>

      {/* Action Button */}
      <button
        type="submit"
        disabled={!selectedFile || isLoading}
        className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:pointer-events-none text-white text-sm font-semibold rounded-xl shadow-lg shadow-indigo-950/50 flex items-center justify-center gap-2 transition focus-visible:outline-2 focus-visible:outline-indigo-400"
      >
        {isLoading ? (
          <>
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            <span>Processing Image & Evaluating Methods...</span>
          </>
        ) : (
          <>
            <ImageIcon className="w-4 h-4" />
            <span>Generate & Compare Captions</span>
          </>
        )}
      </button>
    </form>
  );
};
