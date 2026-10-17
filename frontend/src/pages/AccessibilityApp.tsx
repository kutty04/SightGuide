import React, { useState, useRef, useEffect } from 'react';
import { generateCaption, extractOCR, askQuestion } from '../services/api';
import type { CaptionResponse, OCRResponse, QAResponse, QAHistoryItem } from '../types/api';
import { AppNavbar } from '../components/layout/AppNavbar';
import { AppFooter } from '../components/layout/AppFooter';
import {
  Upload,
  Camera,
  CameraOff,
  SwitchCamera,
  X,
  Volume2,
  FileText,
  Copy,
  Check,
  RefreshCw,
  Sparkles,
  Pause,
  Play,
  RotateCcw,
  VolumeX,
  Eye,
  Sliders,
  Layers,
  ShieldCheck,
  ShieldAlert,
  Image as ImageIcon,
  MessageSquare,
  Send,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Mic,
  MicOff,
} from 'lucide-react';

interface ImageMetadata {
  name: string;
  sizeFormatted: string;
  type: string;
  dimensions: string;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  capability?: string;
  modelName?: string | null;
  isUncertain?: boolean;
  groundedSources?: string[];
  evidenceUsed?: string[];
  limitationNotice?: string;
}

const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB

const SUGGESTED_QUESTIONS = [
  'What are the people wearing?',
  'What colours are visible?',
  'What objects are on the table?',
  'Describe the surroundings.',
  'What does the visible text say?',
  'Is there an object near the person?',
  'What details are difficult to see clearly?',
];

export const AccessibilityApp: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<ImageMetadata | null>(null);

  // Real In-App Camera state
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraFacing, setCameraFacing] = useState<'environment' | 'user'>('environment');
  const [capturedPhotoUrl, setCapturedPhotoUrl] = useState<string | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [isCameraStarting, setIsCameraStarting] = useState<boolean>(false);

  // Advanced algorithm settings (collapsed by default)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [showResearchDetails, setShowResearchDetails] = useState<boolean>(false);
  const [lambdaWeight, setLambdaWeight] = useState<number>(0.5);
  const [decodingStrategy, setDecodingStrategy] = useState<'sample' | 'beam'>('sample');

  // Processing state
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processingStage, setProcessingStage] = useState<string>('');
  const [captionResult, setCaptionResult] = useState<CaptionResponse | null>(null);
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Conversational Q&A state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [isAskingQuestion, setIsAskingQuestion] = useState<boolean>(false);
  const [qaError, setQaError] = useState<string | null>(null);

  // Real image data (base64 data URL) for direct vision-conditioned Q&A
  const [currentImageData, setCurrentImageData] = useState<string | null>(null);

  // Speech Recognition (Voice Input) state
  const [isListening, setIsListening] = useState<boolean>(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);

  // Copy states
  const [copiedCaption, setCopiedCaption] = useState<boolean>(false);
  const [copiedOcr, setCopiedOcr] = useState<boolean>(false);

  // Unified TTS audio state
  const [ttsSupported, setTtsSupported] = useState<boolean>(true);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [currentlySpeakingId, setCurrentlySpeakingId] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Check speech synthesis on mount and clean up on unmount
  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      setTtsSupported(false);
    }
    return () => {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (_) {}
      }
    };
  }, []);

  // Auto-scroll chat to latest message
  useEffect(() => {
    if (chatMessages.length > 0) {
      chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isAskingQuestion]);

  // Clean up camera stream on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
      }
      if (capturedPhotoUrl) {
        URL.revokeObjectURL(capturedPhotoUrl);
      }
    };
  }, [cameraStream, capturedPhotoUrl]);

  // Connect camera stream to video element when stream is ready
  useEffect(() => {
    if (videoRef.current && cameraStream && !capturedPhotoUrl) {
      videoRef.current.srcObject = cameraStream;
      videoRef.current.play().catch((err) => {
        console.warn('Video auto-play interrupted:', err);
      });
    }
  }, [cameraStream, isCameraActive, capturedPhotoUrl]);

  const stopAllSpeech = () => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsPlaying(false);
    setIsPaused(false);
    setCurrentlySpeakingId(null);
  };

  // --- CAMERA METHODS ---
  const startCamera = async (facing: 'environment' | 'user' = cameraFacing) => {
    stopAllSpeech();
    setCameraError(null);
    setErrorMessage(null);
    setIsCameraStarting(true);
    setIsCameraActive(true);

    if (capturedPhotoUrl) {
      URL.revokeObjectURL(capturedPhotoUrl);
      setCapturedPhotoUrl(null);
    }
    setCapturedBlob(null);

    // Stop existing stream if running
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
    }

    // Check MediaDevices support and secure context
    if (typeof window === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError(
        'In-app camera is not supported by your browser or requires a secure context (HTTPS or localhost).'
      );
      setIsCameraStarting(false);
      return;
    }

    try {
      let stream: MediaStream;
      try {
        // Try requested facing mode (ideal: environment for rear camera on mobile)
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facing },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });
      } catch (constraintErr: any) {
        console.warn('Constrained camera request failed, falling back to default video:', constraintErr);
        // Fall back to unconstrained video
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      setCameraStream(stream);
      setCameraFacing(facing);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch((e) => console.error('Video play error:', e));
      }
    } catch (err: any) {
      console.error('Camera acquisition error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError(
          'Camera permission was denied. Please allow camera access in your browser settings or address bar icon.'
        );
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraError('No camera was detected on this device. Please connect a webcam or use file upload.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setCameraError('Camera is currently in use by another application or browser tab.');
      } else {
        setCameraError(`Camera error: ${err.message || 'Unable to access camera.'}`);
      }
    } finally {
      setIsCameraStarting(false);
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (capturedPhotoUrl) {
      URL.revokeObjectURL(capturedPhotoUrl);
      setCapturedPhotoUrl(null);
    }
    setCapturedBlob(null);
    setIsCameraActive(false);
    setCameraError(null);
    setIsCameraStarting(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw the current video frame onto canvas
    ctx.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setCameraError('Failed to capture frame from camera stream.');
          return;
        }
        const url = URL.createObjectURL(blob);
        setCapturedBlob(blob);
        setCapturedPhotoUrl(url);
      },
      'image/jpeg',
      0.92
    );
  };

  const retakePhoto = () => {
    if (capturedPhotoUrl) {
      URL.revokeObjectURL(capturedPhotoUrl);
      setCapturedPhotoUrl(null);
    }
    setCapturedBlob(null);
    if (videoRef.current && cameraStream) {
      videoRef.current.play().catch((e) => console.warn(e));
    }
  };

  const useCapturedPhoto = () => {
    if (!capturedBlob) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const photoFile = new File([capturedBlob], `camera_capture_${timestamp}.jpg`, {
      type: 'image/jpeg',
      lastModified: Date.now(),
    });

    // Send captured photo into the existing image validation and pipeline!
    validateAndSelectFile(photoFile);

    // Stop and release camera tracks
    stopCamera();
  };

  const switchCameraFacing = () => {
    const nextFacing = cameraFacing === 'environment' ? 'user' : 'environment';
    startCamera(nextFacing);
  };

  // --- FILE SELECTION & PIPELINE METHODS ---
  const validateAndSelectFile = (file: File) => {
    stopAllSpeech();
    setErrorMessage(null);

    // Validate type
    if (!ALLOWED_IMAGE_TYPES.includes(file.type.toLowerCase()) && !file.name.match(/\.(jpe?g|png|webp|bmp)$/i)) {
      setErrorMessage(`Unsupported format (${file.type || 'unknown'}). Please upload a JPEG, PNG, WebP, or BMP image.`);
      return;
    }

    // Validate size
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum supported size is 15 MB.`);
      return;
    }

    setSelectedFile(file);
    setCaptionResult(null);
    setOcrResult(null);
    setChatMessages([]);
    setQaError(null);
    setVoiceError(null);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Read base64 data URL for direct vision conditioning
    const reader = new FileReader();
    reader.onload = (e) => {
      setCurrentImageData(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Read dimensions
    const img = new Image();
    img.onload = () => {
      setMetadata({
        name: file.name,
        sizeFormatted: `${(file.size / 1024).toFixed(1)} KB`,
        type: file.type || 'image/jpeg',
        dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`,
      });
    };
    img.src = url;
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMessage(null);
    setCaptionResult(null);
    setOcrResult(null);
    setChatMessages([]);
    stopAllSpeech();

    try {
      setProcessingStage('Extracting visible text & layout with RapidOCR...');
      const ocrRes = await extractOCR(selectedFile);
      setOcrResult(ocrRes);

      setProcessingStage('Analyzing scene with diversity-aware ranking & LLM fusion...');
      const captionRes = await generateCaption(
        selectedFile,
        lambdaWeight,
        'proposed',
        decodingStrategy,
        ocrRes.text,
        ocrRes.normalized_text
      );
      setCaptionResult(captionRes);
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred during image processing.');
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
    }
  };

  const handleReset = () => {
    stopAllSpeech();
    stopCamera();
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch (_) {}
    }
    setIsListening(false);
    setSelectedFile(null);
    setPreviewUrl(null);
    setCurrentImageData(null);
    setMetadata(null);
    setCaptionResult(null);
    setOcrResult(null);
    setChatMessages([]);
    setErrorMessage(null);
    setQaError(null);
    setVoiceError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // TTS Methods
  const speakText = (text: string, identifier: string) => {
    if (!ttsSupported || !text.trim()) return;

    if (isPaused && currentlySpeakingId === identifier) {
      window.speechSynthesis.resume();
      setIsPaused(false);
      setIsPlaying(true);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = playbackSpeed;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      setIsPlaying(true);
      setIsPaused(false);
      setCurrentlySpeakingId(identifier);
    };

    utterance.onend = () => {
      setIsPlaying(false);
      setIsPaused(false);
      setCurrentlySpeakingId(null);
    };

    utterance.onerror = () => {
      setIsPlaying(false);
      setIsPaused(false);
      setCurrentlySpeakingId(null);
    };

    window.speechSynthesis.speak(utterance);
  };

  const handlePauseSpeech = () => {
    if (!ttsSupported || !isPlaying) return;
    window.speechSynthesis.pause();
    setIsPaused(true);
    setIsPlaying(false);
  };

  const handleSpeakAll = () => {
    const parts: string[] = [];
    if (captionResult?.selected_pair?.rank_1_caption) {
      parts.push(`Summary: ${captionResult.selected_pair.rank_1_caption}`);
    }
    if (captionResult?.fusion?.fused_caption) {
      parts.push(`Detailed Description: ${captionResult.fusion.fused_caption}`);
    }
    if (ocrResult?.has_text && ocrResult.text) {
      parts.push(`Visible Text Content: ${ocrResult.text}`);
    } else if (ocrResult && !ocrResult.has_text) {
      parts.push('No readable text detected in this scene.');
    }
    speakText(parts.join('. '), 'master_read_all');
  };

  // Conversational Q&A handler
  const handleAskQuestion = async (questionText?: string) => {
    const q = (questionText || currentQuestion).trim();
    if (!q || isAskingQuestion) return;

    setQaError(null);
    setCurrentQuestion('');

    const userMsgId = `user_${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    setIsAskingQuestion(true);

    try {
      // Build history
      const historyItems: QAHistoryItem[] = chatMessages
        .filter((_, idx) => idx % 2 === 0)
        .slice(-3)
        .map((userItem, i) => {
          const assistantItem = chatMessages[i * 2 + 1];
          return {
            question: userItem.text,
            answer: assistantItem ? assistantItem.text : '',
          };
        });

      // Ensure image_data is guaranteed available from selectedFile
      let activeImageData = currentImageData;
      if (!activeImageData && selectedFile) {
        try {
          activeImageData = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target?.result as string);
            reader.onerror = (e) => reject(e);
            reader.readAsDataURL(selectedFile);
          });
          setCurrentImageData(activeImageData);
        } catch (err) {
          console.warn('Failed to encode image data URL fallback:', err);
        }
      }

      const response: QAResponse = await askQuestion({
        question: q,
        scene_caption: captionResult?.fusion?.fused_caption || captionResult?.selected_pair?.rank_1_caption || null,
        ocr_text: ocrResult?.has_text ? ocrResult.text : null,
        history: historyItems,
        image_data: activeImageData,
      });

      const botMsgId = `assistant_${Date.now()}`;
      const botMsg: ChatMessage = {
        id: botMsgId,
        sender: 'assistant',
        text: response.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        capability: response.capability,
        modelName: response.model_name,
        isUncertain: response.is_uncertain,
        groundedSources: response.grounded_sources,
        evidenceUsed: response.evidence_used,
        limitationNotice: response.limitation_notice,
      };

      setChatMessages((prev) => [...prev, botMsg]);
    } catch (err: any) {
      setQaError(err.message || 'Failed to receive an answer.');
    } finally {
      setIsAskingQuestion(false);
    }
  };

  // Browser Speech Recognition for hands-free voice questions
  const handleToggleVoiceQuestion = () => {
    const SpeechRecognition = typeof window !== 'undefined'
      ? ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)
      : null;

    if (!SpeechRecognition) {
      setVoiceError('Voice input is not supported in this browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (isListening) {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (_) {}
      }
      setIsListening(false);
      return;
    }

    setVoiceError(null);
    stopAllSpeech();

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = true;
      recognition.continuous = false;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((r: any) => r[0].transcript)
          .join('');
        setCurrentQuestion(transcript);
      };

      recognition.onerror = (event: any) => {
        console.warn('Speech recognition error:', event.error);
        setIsListening(false);
        if (event.error === 'not-allowed') {
          setVoiceError('Microphone permission was denied. Please allow microphone access in your browser settings.');
        } else {
          setVoiceError(`Voice input error: ${event.error}`);
        }
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err: any) {
      setIsListening(false);
      setVoiceError(`Could not start voice recognition: ${err.message}`);
    }
  };

  const handleCopyCaption = async () => {
    if (!captionResult?.fusion?.fused_caption) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(captionResult.fusion.fused_caption);
      }
    } catch {
      try {
        const textarea = document.createElement('textarea');
        textarea.value = captionResult.fusion.fused_caption;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      } catch (_) {}
    }
    setCopiedCaption(true);
    setTimeout(() => setCopiedCaption(false), 2000);
  };

  const handleCopyOcr = async () => {
    if (!ocrResult?.text) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(ocrResult.text);
      }
    } catch {
      try {
        const textarea = document.createElement('textarea');
        textarea.value = ocrResult.text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      } catch (_) {}
    }
    setCopiedOcr(true);
    setTimeout(() => setCopiedOcr(false), 2000);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-[#F7F6F2] flex flex-col font-sans selection:bg-[#D4FF00] selection:text-black">
      <AppNavbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Section 1: Page Header & Status */}
        <header className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold bg-[#D4FF00]/10 border border-[#D4FF00]/40 px-2.5 py-0.5">
                  01 // ACCESSIBLE COGNITIVE VISION
                </span>
                <span className="text-[10px] font-mono text-[#D4FF00] bg-black border border-white/20 px-2 py-0.5 flex items-center gap-1.5">
                  <ShieldCheck className="w-3 h-3 text-[#D4FF00]" />
                  GROUNDED EVIDENCE
                </span>
              </div>
              <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight mt-2">
                ACCESSIBILITY ASSISTANT &amp; CONVERSATIONAL VISION
              </h1>
              <p className="text-sm text-zinc-400 max-w-3xl mt-1 leading-relaxed font-sans">
                Open your device camera or upload an image to synthesize a factual scene narrative, extract visible signage and typography, listen via zero-latency speech synthesis, and ask grounded follow-up questions.
              </p>
            </div>

            {selectedFile && (
              <button
                type="button"
                onClick={handleReset}
                className="flex items-center gap-2 px-3.5 py-2 bg-black hover:bg-white/10 text-zinc-300 hover:text-white border border-white/20 hover:border-[#D4FF00] text-xs font-mono tracking-wider uppercase transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                aria-label="Upload a new image and reset session"
              >
                <RefreshCw className="w-3.5 h-3.5 text-[#D4FF00]" />
                <span>NEW SESSION</span>
              </button>
            )}
          </div>
        </header>

        {/* Global Error Notification */}
        {errorMessage && (
          <div
            role="alert"
            className="p-4 bg-red-950/70 border border-red-700 text-red-200 text-xs font-mono flex items-center gap-3 shadow-lg"
          >
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400" />
            <div className="flex-1">
              <strong className="font-bold block uppercase tracking-wider">NOTIFICATION:</strong>
              <span>{errorMessage}</span>
            </div>
          </div>
        )}

        {/* Main Grid: Left Column (Image & Camera) vs Right Column (Results & Conversation) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* LEFT COLUMN: Input Options, Camera Viewfinder, Preview, and Settings */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-[#0c0c0c] border border-white/10 p-6 space-y-5 shadow-xl reticle-bracket">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00] flex items-center gap-2">
                  <ImageIcon className="w-3.5 h-3.5 text-[#D4FF00]" />
                  <span>SOURCE IMAGE / OPTICAL INPUT</span>
                </h2>
                {isCameraActive && (
                  <span className="text-[10px] font-mono text-[#D4FF00] bg-black border border-[#D4FF00] px-2 py-0.5 flex items-center gap-1.5 animate-pulse">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#D4FF00]" />
                    <span>RETICLE ACTIVE</span>
                  </span>
                )}
              </div>

              {/* Hidden file input for desktop upload */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                aria-label="Upload image from file system"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    validateAndSelectFile(e.target.files[0]);
                  }
                }}
              />

              {/* Hidden canvas for video frame capture */}
              <canvas ref={canvasRef} className="hidden" aria-hidden="true" />

              {/* VIEW STATE 1: REAL IN-APP CAMERA ACTIVE */}
              {isCameraActive ? (
                <div className="space-y-4">
                  {/* Camera Card Container */}
                  <div className="relative overflow-hidden border border-white/20 bg-black min-h-[280px] flex flex-col items-center justify-center reticle-bracket">
                    {cameraError ? (
                      <div className="p-6 text-center space-y-3">
                        <CameraOff className="w-10 h-10 mx-auto text-red-400" />
                        <h3 className="text-sm font-mono font-bold text-red-200 uppercase">Camera Unavailable</h3>
                        <p className="text-xs text-red-300/90 max-w-xs mx-auto leading-relaxed font-sans">
                          {cameraError}
                        </p>
                        <div className="flex justify-center gap-2 pt-2">
                          <button
                            type="button"
                            onClick={() => startCamera()}
                            className="px-3.5 py-1.5 bg-[#D4FF00] text-black text-xs font-mono font-bold uppercase transition"
                          >
                            Try Again
                          </button>
                          <button
                            type="button"
                            onClick={stopCamera}
                            className="px-3.5 py-1.5 bg-black border border-white/20 text-xs font-mono text-zinc-300 hover:text-white uppercase transition"
                          >
                            Back to Upload
                          </button>
                        </div>
                      </div>
                    ) : capturedPhotoUrl ? (
                      /* Snapshot Freeze & Review */
                      <div className="relative w-full">
                        <img
                          src={capturedPhotoUrl}
                          alt="Captured frame preview"
                          className="w-full h-64 sm:h-72 object-contain bg-black mx-auto"
                        />
                        <div className="absolute top-2 left-2 bg-black/90 px-2 py-0.5 text-[10px] font-mono text-[#D4FF00] border border-[#D4FF00] tracking-wider uppercase">
                          Photo Captured
                        </div>
                      </div>
                    ) : (
                      /* Live Camera Feed */
                      <div className="relative w-full h-64 sm:h-72 bg-black flex items-center justify-center overflow-hidden">
                        {isCameraStarting && (
                          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/90 text-[#D4FF00] gap-2 text-xs font-mono">
                            <RefreshCw className="w-5 h-5 animate-spin" />
                            <span className="tracking-widest uppercase">Requesting camera stream...</span>
                          </div>
                        )}
                        <video
                          ref={videoRef}
                          playsInline
                          autoPlay
                          muted
                          className="w-full h-full object-cover"
                          aria-label="Live camera viewfinder stream"
                        />
                        {/* Optical HUD Crosshairs Overlay */}
                        <div className="pointer-events-none absolute inset-0 border border-white/10 z-10 flex items-center justify-center">
                          <div className="w-16 h-16 border border-[#D4FF00]/40 rounded-full flex items-center justify-center">
                            <div className="w-2 h-2 bg-[#D4FF00] rounded-full animate-ping" />
                          </div>
                          <div className="absolute top-3 left-3 text-[9px] font-mono text-[#D4FF00]/70 uppercase tracking-widest">
                            [OPTICAL FEED // 1080P]
                          </div>
                        </div>

                        <div className="absolute top-2 right-2 flex items-center gap-1.5 z-20">
                          <button
                            type="button"
                            onClick={switchCameraFacing}
                            className="p-1.5 bg-black/90 hover:bg-white/10 text-white border border-white/20 hover:border-[#D4FF00] text-xs transition"
                            title="Flip camera (front / rear)"
                            aria-label="Switch camera between front and rear"
                          >
                            <SwitchCamera className="w-4 h-4" />
                          </button>
                          <button
                            type="button"
                            onClick={stopCamera}
                            className="p-1.5 bg-black/90 hover:bg-white/10 text-white border border-white/20 hover:border-[#D4FF00] text-xs transition"
                            title="Close camera"
                            aria-label="Close camera and return to upload"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Camera Action Controls */}
                  {!cameraError && (
                    <div>
                      {capturedPhotoUrl ? (
                        <div className="grid grid-cols-2 gap-2">
                          <button
                            type="button"
                            onClick={retakePhoto}
                            className="flex items-center justify-center gap-2 py-2.5 px-3 bg-black hover:bg-white/10 text-zinc-200 text-xs font-mono uppercase tracking-wider border border-white/20 transition"
                            aria-label="Retake photo"
                          >
                            <RotateCcw className="w-3.5 h-3.5" />
                            <span>Retake</span>
                          </button>
                          <button
                            type="button"
                            onClick={useCapturedPhoto}
                            className="flex items-center justify-center gap-2 py-2.5 px-3 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-wider shadow-md transition"
                            aria-label="Use captured photo for analysis"
                          >
                            <Check className="w-3.5 h-3.5" />
                            <span>Use Frame</span>
                          </button>
                        </div>
                      ) : (
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={capturePhoto}
                            disabled={isCameraStarting}
                            className="flex-1 flex items-center justify-center gap-2 py-3 px-4 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-wider disabled:opacity-50 transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                            aria-label="Snap photo from camera"
                          >
                            <Camera className="w-4 h-4" />
                            <span>Capture Frame</span>
                          </button>
                          <button
                            type="button"
                            onClick={stopCamera}
                            className="px-3.5 py-3 bg-black hover:bg-white/10 text-zinc-300 text-xs font-mono uppercase tracking-wider border border-white/20 transition"
                            aria-label="Cancel camera and return"
                          >
                            Cancel
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : !previewUrl ? (
                /* VIEW STATE 2: NO IMAGE YET -> CHOOSE OPEN CAMERA OR BROWSE FILES */
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Real Open Camera Button */}
                    <button
                      type="button"
                      onClick={() => startCamera('environment')}
                      className="flex flex-col items-center justify-center p-5 bg-black border border-white/20 hover:border-[#D4FF00] text-white transition group focus-visible:outline-2 focus-visible:outline-[#D4FF00] reticle-bracket"
                      aria-label="Open camera to capture a live photo"
                    >
                      <div className="w-10 h-10 border border-white/20 bg-[#0c0c0c] flex items-center justify-center mb-2 text-[#D4FF00] group-hover:bg-[#D4FF00] group-hover:text-black transition-colors">
                        <Camera className="w-5 h-5" />
                      </div>
                      <span className="text-xs font-mono font-bold uppercase tracking-wider">Open Camera</span>
                      <span className="text-[10px] text-zinc-500 font-mono mt-0.5">Live Viewfinder</span>
                    </button>

                    {/* Standard File Upload Button */}
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex flex-col items-center justify-center p-5 bg-black border border-white/20 hover:border-[#D4FF00] text-white transition group focus-visible:outline-2 focus-visible:outline-[#D4FF00] reticle-bracket"
                      aria-label="Browse and upload existing image file"
                    >
                      <div className="w-10 h-10 border border-white/20 bg-[#0c0c0c] flex items-center justify-center mb-2 text-zinc-300 group-hover:bg-[#D4FF00] group-hover:text-black transition-colors">
                        <Upload className="w-5 h-5" />
                      </div>
                      <span className="text-xs font-mono font-bold uppercase tracking-wider">Browse Files</span>
                      <span className="text-[10px] text-zinc-500 font-mono mt-0.5">JPEG / PNG / WebP</span>
                    </button>
                  </div>

                  {/* Drag and Drop Zone */}
                  <div
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-white/20 hover:border-[#D4FF00] p-6 text-center cursor-pointer transition-colors bg-black group"
                    role="button"
                    tabIndex={0}
                    aria-label="Drop image files here"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        fileInputRef.current?.click();
                      }
                    }}
                  >
                    <p className="text-xs text-zinc-400 font-mono uppercase tracking-wider">
                      Or drag and drop image file here (up to 15MB)
                    </p>
                  </div>
                </div>
              ) : (
                /* VIEW STATE 3: IMAGE LOADED AND READY FOR ANALYSIS */
                <div className="space-y-4">
                  {/* Image Preview Container */}
                  <div className="relative overflow-hidden border border-white/20 bg-black max-h-72 flex items-center justify-center reticle-bracket">
                    <img
                      src={previewUrl}
                      alt="Uploaded scene preview for accessibility description"
                      className="max-h-72 w-auto object-contain mx-auto"
                    />
                  </div>

                  {/* Metadata Summary Card */}
                  {metadata && (
                    <div className="grid grid-cols-2 gap-2 text-[11px] bg-black p-3.5 border border-white/10 font-mono text-zinc-300">
                      <div>
                        <span className="text-zinc-500 uppercase block text-[9px]">Dimensions:</span>
                        <span className="font-bold text-white">{metadata.dimensions}</span>
                      </div>
                      <div>
                        <span className="text-zinc-500 uppercase block text-[9px]">File Size:</span>
                        <span className="font-bold text-white">{metadata.sizeFormatted}</span>
                      </div>
                      <div className="col-span-2 truncate">
                        <span className="text-zinc-500 uppercase block text-[9px]">File Name:</span>
                        <span className="font-bold text-zinc-200 truncate">{metadata.name}</span>
                      </div>
                    </div>
                  )}

                  {/* Action row to change image */}
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => startCamera('environment')}
                      className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-black hover:bg-white/10 text-zinc-300 text-xs font-mono uppercase tracking-wider border border-white/20 transition"
                      aria-label="Open camera to take a replacement photo"
                    >
                      <Camera className="w-3.5 h-3.5 text-[#D4FF00]" />
                      <span>Take Photo</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 bg-black hover:bg-white/10 text-zinc-300 text-xs font-mono uppercase tracking-wider border border-white/20 transition"
                      aria-label="Upload a replacement image from files"
                    >
                      <Upload className="w-3.5 h-3.5 text-zinc-400" />
                      <span>Change File</span>
                    </button>
                  </div>

                  {/* Analyze Scene Button */}
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    disabled={isProcessing}
                    className="w-full flex items-center justify-center gap-2 py-3.5 px-4 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-widest shadow-[0_0_20px_rgba(212,255,0,0.2)] disabled:opacity-50 disabled:pointer-events-none transition-all focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                    aria-label="Describe scene and extract text"
                  >
                    {isProcessing ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-black" />
                        <span>{processingStage || 'PROCESSING PIPELINE...'}</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4 text-black" />
                        <span>DESCRIBE SCENE &amp; EXTRACT TEXT</span>
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Advanced Controls Toggle (MMR Lambda & BLIP Decoding) */}
              <div className="border-t border-white/10 pt-4">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center justify-between w-full text-xs font-mono text-zinc-400 hover:text-white transition focus-visible:outline-none"
                  aria-expanded={showAdvanced}
                >
                  <span className="flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-[#D4FF00]" />
                    <span className="uppercase tracking-wider">Algorithm Settings (MMR &amp; Decoding)</span>
                  </span>
                  <span className="font-mono text-[10px] text-[#D4FF00]">
                    {showAdvanced ? '[ HIDE ]' : '[ CONFIGURE ]'}
                  </span>
                </button>

                {showAdvanced && (
                  <div className="mt-3 space-y-4 bg-black p-3.5 border border-white/10 text-xs font-mono">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label htmlFor="lambda-slider" className="text-zinc-300 uppercase tracking-wider text-[11px]">
                          MMR Lambda Weight (λ):
                        </label>
                        <span className="font-mono font-bold text-[#D4FF00]">{lambdaWeight.toFixed(2)}</span>
                      </div>
                      <input
                        id="lambda-slider"
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={lambdaWeight}
                        onChange={(e) => setLambdaWeight(parseFloat(e.target.value))}
                        className="w-full accent-[#D4FF00] cursor-pointer"
                        aria-valuemin={0.0}
                        aria-valuemax={1.0}
                        aria-valuenow={lambdaWeight}
                      />
                      <div className="flex justify-between text-[10px] text-zinc-500 font-mono mt-0.5">
                        <span>Max Diversity (0.0)</span>
                        <span className="text-[#D4FF00]">Balanced (0.5)</span>
                        <span>Max Grounding (1.0)</span>
                      </div>
                    </div>

                    <div>
                      <label className="text-zinc-300 uppercase tracking-wider text-[11px] block mb-1.5">
                        BLIP Decoding Strategy:
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => setDecodingStrategy('sample')}
                          className={`py-1.5 px-3 text-xs font-mono uppercase tracking-wider border text-center transition ${
                            decodingStrategy === 'sample'
                              ? 'bg-[#D4FF00] border-[#D4FF00] text-black font-bold'
                              : 'bg-black border-white/20 text-zinc-400 hover:text-white'
                          }`}
                        >
                          Sampling (Default)
                        </button>
                        <button
                          type="button"
                          onClick={() => setDecodingStrategy('beam')}
                          className={`py-1.5 px-3 text-xs font-mono uppercase tracking-wider border text-center transition ${
                            decodingStrategy === 'beam'
                              ? 'bg-[#D4FF00] border-[#D4FF00] text-black font-bold'
                              : 'bg-black border-white/20 text-zinc-400 hover:text-white'
                          }`}
                        >
                          Beam Search
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Spoken Master Bar, Descriptions, OCR, and Conversational Q&A */}
          <div className="lg:col-span-7 space-y-6">
            {/* Audio Reader Controls Bar */}
            {(captionResult || ocrResult) && (
              <section
                aria-label="Audio Reader for Screen Readers and Blind Users"
                className="bg-[#0c0c0c] border border-[#D4FF00] p-5 space-y-4 shadow-[0_0_20px_rgba(212,255,0,0.15)] reticle-bracket"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 border border-[#D4FF00] bg-black flex items-center justify-center text-[#D4FF00]">
                      <Volume2 className="w-4 h-4" />
                    </div>
                    <div>
                      <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
                        AUDITORY SYNTHESIZER (WEB SPEECH API)
                      </h2>
                      <p className="text-xs text-zinc-300 font-mono mt-0.5" aria-live="polite">
                        {isPlaying
                          ? `[STATUS: SPEAKING - ${
                              currentlySpeakingId === 'master_read_all'
                                ? 'Full Scene & Visible Text'
                                : currentlySpeakingId === 'caption_concise'
                                ? 'Concise Summary'
                                : currentlySpeakingId === 'caption_detailed'
                                ? 'Detailed Description'
                                : currentlySpeakingId === 'ocr_text'
                                ? 'Visible Text'
                                : 'Question Answer'
                            }]`
                          : isPaused
                          ? '[STATUS: PLAYBACK PAUSED]'
                          : '[STATUS: READY TO READ ALOUD]'}
                      </p>
                    </div>
                  </div>

                  {/* Playback speed selector */}
                  <div className="flex items-center gap-1.5 bg-black border border-white/20 px-2.5 py-1 text-xs font-mono">
                    <label htmlFor="tts-speed" className="text-zinc-400 uppercase tracking-wider text-[10px]">
                      Speed:
                    </label>
                    <select
                      id="tts-speed"
                      value={playbackSpeed}
                      onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                      className="bg-transparent text-[#D4FF00] font-mono font-bold focus:outline-none cursor-pointer"
                    >
                      <option value="0.75" className="bg-black text-white">0.75x</option>
                      <option value="1.0" className="bg-black text-white">1.0x</option>
                      <option value="1.25" className="bg-black text-white">1.25x</option>
                      <option value="1.5" className="bg-black text-white">1.5x</option>
                    </select>
                  </div>
                </div>

                {/* Speech Control Buttons */}
                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={handleSpeakAll}
                    disabled={!ttsSupported}
                    className="flex items-center gap-2 px-4 py-2 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-wider shadow-md transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                    aria-label="Read complete scene description and visible text aloud"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Read Full Scene (Summary + Text)</span>
                  </button>

                  {isPlaying && (
                    <button
                      type="button"
                      onClick={handlePauseSpeech}
                      className="flex items-center gap-1.5 px-3 py-2 bg-amber-400 hover:bg-amber-300 text-black font-mono font-bold text-xs uppercase tracking-wider shadow transition focus-visible:outline-2 focus-visible:outline-amber-300"
                      aria-label="Pause speech"
                    >
                      <Pause className="w-3.5 h-3.5 fill-current" />
                      <span>Pause</span>
                    </button>
                  )}

                  {(isPlaying || isPaused) && (
                    <button
                      type="button"
                      onClick={stopAllSpeech}
                      className="flex items-center gap-1.5 px-3 py-2 bg-black hover:bg-white/10 text-zinc-300 font-mono text-xs uppercase tracking-wider border border-white/20 transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                      aria-label="Stop speech playback"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Stop</span>
                    </button>
                  )}

                  {!ttsSupported && (
                    <div className="text-xs font-mono text-amber-400 flex items-center gap-1.5">
                      <VolumeX className="w-4 h-4" />
                      <span>Speech synthesis is unavailable in this browser.</span>
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* Scene Understanding Card */}
            {captionResult && (
              <section
                aria-label="Visual Scene Understanding"
                className="bg-[#0c0c0c] border border-white/10 p-6 space-y-5 shadow-xl reticle-bracket"
              >
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4 text-[#D4FF00]" />
                    <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
                      VISUAL SCENE UNDERSTANDING
                    </h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => speakText(captionResult.fusion.fused_caption, 'caption_detailed')}
                      className="flex items-center gap-1 text-xs font-mono text-zinc-300 hover:text-white px-2 py-1 bg-black border border-white/20 hover:border-[#D4FF00] transition"
                      title="Read detailed caption aloud"
                      aria-label="Read detailed description aloud"
                    >
                      <Volume2 className="w-3 h-3 text-[#D4FF00]" />
                      <span>READ SCENE</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleCopyCaption}
                      className="flex items-center gap-1 text-xs font-mono text-zinc-300 hover:text-white px-2 py-1 bg-black border border-white/20 hover:border-[#D4FF00] transition"
                      title="Copy caption to clipboard"
                      aria-label="Copy description to clipboard"
                    >
                      {copiedCaption ? <Check className="w-3 h-3 text-[#D4FF00]" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedCaption ? 'COPIED' : 'COPY'}</span>
                    </button>
                  </div>
                </div>

                {/* Concise Summary vs Detailed Description */}
                <div className="space-y-4">
                  {/* Concise Summary (Visual Anchor) */}
                  <div className="bg-black border border-white/10 p-4 space-y-1.5">
                    <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold">
                      <span>Concise Summary (Anchor c₁*)</span>
                      <button
                        type="button"
                        onClick={() => speakText(captionResult.selected_pair.rank_1_caption, 'caption_concise')}
                        className="hover:underline flex items-center gap-1 text-zinc-400 hover:text-[#D4FF00]"
                        aria-label="Read concise summary"
                      >
                        <Volume2 className="w-3 h-3" />
                        <span>LISTEN</span>
                      </button>
                    </div>
                    <p className="text-sm font-semibold text-white leading-relaxed font-sans">
                      {captionResult.selected_pair.rank_1_caption}
                    </p>
                  </div>

                  {/* Detailed Grounded Description (LLM Fusion) */}
                  <div className="bg-black border-l-2 border-[#D4FF00] p-4 space-y-2 border-y border-r border-white/10">
                    <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold">
                      <span>Detailed Grounded Narrative</span>
                      <span className="text-[10px] text-zinc-500 font-mono">CONSERVATIVE V3 FUSION</span>
                    </div>
                    <p
                      tabIndex={0}
                      className="text-base sm:text-lg font-medium text-white italic leading-relaxed selection:bg-[#D4FF00] selection:text-black focus:outline-none focus:ring-1 focus:ring-[#D4FF00] p-1 font-sans"
                    >
                      &ldquo;{captionResult.fusion.fused_caption}&rdquo;
                    </p>
                  </div>
                </div>

                {/* Collapsible Research Details */}
                <div className="border-t border-white/10 pt-3">
                  <button
                    type="button"
                    onClick={() => setShowResearchDetails(!showResearchDetails)}
                    className="flex items-center justify-between w-full text-xs font-mono text-zinc-400 hover:text-white transition focus-visible:outline-none"
                    aria-expanded={showResearchDetails}
                  >
                    <span className="flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5 text-[#D4FF00]" />
                      <span className="uppercase tracking-wider">Candidate Breakdown &amp; Log-Likelihood Scores</span>
                    </span>
                    <span className="font-mono text-[10px] text-[#D4FF00] flex items-center gap-0.5">
                      {showResearchDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                      <span>{showResearchDetails ? '[ HIDE ]' : '[ INSPECT ]'}</span>
                    </span>
                  </button>

                  {showResearchDetails && (
                    <div className="mt-3 space-y-3 bg-black p-4 border border-white/10 text-xs">
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div className="p-3 bg-[#0c0c0c] border border-white/10 space-y-1.5">
                          <span className="text-[#D4FF00] font-mono font-bold uppercase tracking-wider block text-[10px]">
                            Anchor Candidate (c₁*):
                          </span>
                          <p className="text-zinc-200 italic font-sans">&ldquo;{captionResult.selected_pair.rank_1_caption}&rdquo;</p>
                          <span className="font-mono text-[10px] text-zinc-400 bg-black px-2 py-0.5 border border-white/10 inline-block">
                            LL Score: {captionResult.selected_pair.rank_1_clip_score.toFixed(3)}
                          </span>
                        </div>

                        <div className="p-3 bg-[#0c0c0c] border border-white/10 space-y-1.5">
                          <span className="text-[#D4FF00] font-mono font-bold uppercase tracking-wider block text-[10px]">
                            MMR Secondary Detail (c₂*):
                          </span>
                          <p className="text-zinc-200 italic font-sans">&ldquo;{captionResult.selected_pair.rank_2_caption}&rdquo;</p>
                          <span className="font-mono text-[10px] text-zinc-400 bg-black px-2 py-0.5 border border-white/10 inline-block">
                            LL Score: {captionResult.selected_pair.rank_2_clip_score.toFixed(3)}
                          </span>
                        </div>
                      </div>

                      <p className="text-[10px] text-zinc-500 font-mono leading-relaxed">
                        Note: Scores indicate negative sequence cross-entropy loss (higher is better). They represent relative model likelihood, not calibrated probabilities.
                      </p>
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* OCR Extracted Text Card */}
            {ocrResult && (
              <section
                aria-label="Visible Scene Text (OCR)"
                className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 shadow-xl reticle-bracket"
              >
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-[#D4FF00]" />
                    <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
                      VISIBLE TEXT EXTRACTION (RAPIDOCR)
                    </h2>
                  </div>

                  {ocrResult.has_text && (
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => speakText(ocrResult.text, 'ocr_text')}
                        className="flex items-center gap-1 text-xs font-mono text-zinc-300 hover:text-white px-2 py-1 bg-black border border-white/20 hover:border-[#D4FF00] transition"
                        title="Read visible text aloud"
                        aria-label="Read visible text aloud"
                      >
                        <Volume2 className="w-3 h-3 text-[#D4FF00]" />
                        <span>READ TEXT</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleCopyOcr}
                        className="flex items-center gap-1 text-xs font-mono text-zinc-300 hover:text-white px-2 py-1 bg-black border border-white/20 hover:border-[#D4FF00] transition"
                        title="Copy detected text"
                        aria-label="Copy detected text to clipboard"
                      >
                        {copiedOcr ? <Check className="w-3 h-3 text-[#D4FF00]" /> : <Copy className="w-3 h-3" />}
                        <span>{copiedOcr ? 'COPIED' : 'COPY'}</span>
                      </button>
                    </div>
                  )}
                </div>

                {ocrResult.has_text ? (
                  <div className="space-y-4">
                    {/* Formatted reading order paragraph display */}
                    <div className="bg-black border border-white/10 p-4">
                      <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold block mb-1.5">
                        Reading Order Text Content
                      </span>
                      <p
                        tabIndex={0}
                        className="text-sm font-mono text-zinc-200 leading-relaxed whitespace-pre-line focus:outline-none focus:ring-1 focus:ring-[#D4FF00] p-1"
                      >
                        {ocrResult.text}
                      </p>
                    </div>

                    {/* Detected Lines Breakdown */}
                    {ocrResult.lines && ocrResult.lines.length > 0 && (
                      <div className="space-y-2">
                        <span className="text-[11px] font-mono uppercase tracking-wider text-zinc-400 block">
                          Detected Layout ({ocrResult.lines.length} lines ordered top-to-bottom):
                        </span>
                        <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
                          {ocrResult.lines.map((line, idx) => (
                            <div
                              key={idx}
                              className="flex items-center justify-between bg-black border border-white/10 px-3 py-1.5 text-xs font-mono"
                            >
                              <div className="flex items-center gap-2 truncate mr-2">
                                <span className="font-mono text-[10px] text-zinc-500 w-5">
                                  #{line.reading_order || idx + 1}
                                </span>
                                <span className="text-zinc-200 truncate">{line.text}</span>
                              </div>
                              <span className="font-mono text-[10px] text-[#D4FF00] bg-[#D4FF00]/10 px-2 py-0.5 border border-[#D4FF00]/40 flex-shrink-0">
                                {(line.confidence * 100).toFixed(0)}% conf
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <p className="text-[10px] text-zinc-500 font-mono">
                      Notice: Confidence percentages represent OCR model recognition certainty, not human-verified correctness.
                    </p>
                  </div>
                ) : (
                  <div className="bg-black border border-white/10 p-6 text-center text-zinc-400 space-y-1.5">
                    <FileText className="w-8 h-8 mx-auto text-zinc-600 mb-2" />
                    <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-300">
                      No Readable Text Detected
                    </h3>
                    <p className="text-[11px] text-zinc-500 font-sans">
                      RapidOCR scanned the image pixels and found no confident printed signage, documents, or labels.
                    </p>
                  </div>
                )}
              </section>
            )}

            {/* Conversational Visual Q&A Card */}
            {(captionResult || ocrResult) && (
              <section
                aria-label="Conversational Visual Assistant"
                className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 shadow-xl reticle-bracket"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-[#D4FF00]" />
                    <h2 className="text-xs font-mono font-bold uppercase tracking-widest text-[#D4FF00]">
                      CONVERSATIONAL VISUAL Q&amp;A
                    </h2>
                  </div>
                  {currentImageData ? (
                    <span className="text-[10px] font-mono text-[#D4FF00] bg-[#D4FF00]/10 border border-[#D4FF00]/40 px-2 py-0.5 flex items-center gap-1.5">
                      <Eye className="w-3 h-3 text-[#D4FF00]" />
                      VISION-CONDITIONED VQA
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono text-amber-400 bg-amber-950/80 border border-amber-800 px-2 py-0.5 flex items-center gap-1">
                      <HelpCircle className="w-3 h-3 text-amber-400" />
                      CONTEXT-GROUNDED Q&amp;A
                    </span>
                  )}
                </div>

                <p className="text-xs text-zinc-400 font-sans leading-relaxed">
                  Ask questions regarding visible clothing, colors, objects, spatial arrangements, surroundings, or visible signage. Responses are grounded strictly in image pixels.
                </p>

                {/* Suggested Question Chips */}
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {SUGGESTED_QUESTIONS.map((sq, idx) => (
                    <button
                      key={idx}
                      type="button"
                      disabled={isAskingQuestion}
                      onClick={() => handleAskQuestion(sq)}
                      className="text-[11px] font-mono px-2.5 py-1 bg-black hover:bg-white/10 text-zinc-300 hover:text-white border border-white/20 hover:border-[#D4FF00] transition disabled:opacity-40 text-left focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                    >
                      {sq}
                    </button>
                  ))}
                </div>

                {/* Conversation Thread */}
                <div className="space-y-3 min-h-[140px] max-h-80 overflow-y-auto p-4 bg-black border border-white/10" role="log" aria-live="polite">
                  {chatMessages.length === 0 ? (
                    <div className="text-center py-8 text-zinc-500 text-xs font-mono">
                      <HelpCircle className="w-6 h-6 mx-auto mb-2 text-[#D4FF00]/50" />
                      <span>NO QUERIES RECORDED. TYPE OR SPEAK A QUESTION BELOW.</span>
                    </div>
                  ) : (
                    chatMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex flex-col space-y-1.5 ${
                          msg.sender === 'user' ? 'items-end' : 'items-start'
                        }`}
                      >
                        <div
                          className={`max-w-[85%] p-3.5 text-xs leading-relaxed ${
                            msg.sender === 'user'
                              ? 'bg-[#D4FF00] text-black font-sans font-medium'
                              : 'bg-[#0c0c0c] border border-white/15 text-zinc-200 font-sans'
                          }`}
                        >
                          <div className={`flex items-center justify-between gap-2 mb-1.5 text-[9px] font-mono uppercase tracking-wider ${
                            msg.sender === 'user' ? 'text-black/70' : 'text-zinc-400'
                          }`}>
                            <span>{msg.sender === 'user' ? 'User' : 'SightGuide Assistant'}</span>
                            <span>{msg.timestamp}</span>
                          </div>
                          <p>{msg.text}</p>

                          {/* Assistant badges & spoken audio trigger */}
                          {msg.sender === 'assistant' && (
                            <div className="mt-2.5 pt-2 border-t border-white/10 flex flex-wrap items-center justify-between gap-2">
                              <div className="flex flex-wrap items-center gap-1">
                                {msg.capability === 'VISION_CONDITIONED' ? (
                                  <span className="text-[9px] font-mono text-[#D4FF00] bg-black px-1.5 py-0.5 border border-[#D4FF00]/40 flex items-center gap-1">
                                    <Eye className="w-2.5 h-2.5 text-[#D4FF00]" />
                                    VISUAL EVIDENCE
                                  </span>
                                ) : (
                                  <span className="text-[9px] font-mono text-amber-300 bg-black px-1.5 py-0.5 border border-amber-800 flex items-center gap-1">
                                    <FileText className="w-2.5 h-2.5 text-amber-400" />
                                    CONTEXT GROUNDED
                                  </span>
                                )}

                                {msg.isUncertain && (
                                  <span className="text-[9px] font-mono text-violet-300 bg-black px-1.5 py-0.5 border border-violet-800 flex items-center gap-1">
                                    <ShieldAlert className="w-2.5 h-2.5 text-violet-400" />
                                    OUT OF FRAME
                                  </span>
                                )}

                                {msg.groundedSources?.map((src, i) => (
                                  <span
                                    key={i}
                                    className="text-[9px] font-mono text-zinc-400 bg-black px-1 py-0.5 border border-white/10"
                                  >
                                    {src === 'visual_image'
                                      ? 'PIXELS'
                                      : src === 'scene_caption'
                                      ? 'SCENE'
                                      : 'OCR'}
                                  </span>
                                ))}
                              </div>

                              <button
                                type="button"
                                onClick={() => speakText(msg.text, msg.id)}
                                className={`text-[10px] flex items-center gap-1 font-mono uppercase tracking-wider transition ${
                                  currentlySpeakingId === msg.id && isPlaying
                                    ? 'text-[#D4FF00] font-bold underline animate-pulse'
                                    : 'text-[#D4FF00] hover:underline'
                                }`}
                                aria-label="Listen to this answer aloud"
                              >
                                <Volume2 className="w-3 h-3" />
                                <span>{currentlySpeakingId === msg.id && isPlaying ? 'PLAYING' : 'LISTEN'}</span>
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    ))
                  )}

                  {isAskingQuestion && (
                    <div className="flex items-center gap-2 text-xs text-[#D4FF00] font-mono py-2">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span className="tracking-widest uppercase">Inspecting visual evidence and formulating answer...</span>
                    </div>
                  )}

                  <div ref={chatBottomRef} />
                </div>

                {qaError && (
                  <div role="alert" className="text-xs font-mono text-red-400 bg-red-950/60 p-2.5 border border-red-800 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span>{qaError}</span>
                  </div>
                )}

                {voiceError && (
                  <div role="alert" className="text-xs font-mono text-amber-300 bg-amber-950/60 p-2.5 border border-amber-800 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span>{voiceError}</span>
                  </div>
                )}

                {/* Question Input Form with Voice Button */}
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleAskQuestion();
                  }}
                  className="flex gap-2"
                >
                  <input
                    type="text"
                    value={currentQuestion}
                    onChange={(e) => setCurrentQuestion(e.target.value)}
                    placeholder="Ask a question about this image (e.g. 'What are the people wearing?')..."
                    disabled={isAskingQuestion}
                    className="flex-1 bg-black border border-white/20 focus:border-[#D4FF00] px-3.5 py-2.5 text-xs text-white placeholder-zinc-500 focus:outline-none disabled:opacity-50 font-sans"
                    aria-label="Type your question about this image"
                  />

                  {/* Hands-Free Voice Question Control */}
                  <button
                    type="button"
                    onClick={handleToggleVoiceQuestion}
                    disabled={isAskingQuestion}
                    className={`px-3 py-2.5 border text-xs font-mono uppercase tracking-wider flex items-center gap-1.5 transition focus-visible:outline-2 focus-visible:outline-[#D4FF00] ${
                      isListening
                        ? 'bg-red-600 text-white border-red-500 animate-pulse'
                        : 'bg-black hover:bg-white/10 text-zinc-300 hover:text-white border-white/20'
                    }`}
                    aria-label={isListening ? 'Stop listening to voice question' : 'Speak question with voice microphone'}
                    title={isListening ? 'Stop voice listening' : 'Ask by speaking'}
                  >
                    {isListening ? <MicOff className="w-3.5 h-3.5 text-white" /> : <Mic className="w-3.5 h-3.5 text-[#D4FF00]" />}
                    <span className="hidden sm:inline">{isListening ? 'LISTENING...' : 'VOICE'}</span>
                  </button>

                  <button
                    type="submit"
                    disabled={isAskingQuestion || !currentQuestion.trim()}
                    className="px-4 py-2.5 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-wider flex items-center gap-1.5 disabled:opacity-40 disabled:pointer-events-none transition focus-visible:outline-2 focus-visible:outline-[#D4FF00]"
                    aria-label="Send question"
                  >
                    <span>ASK</span>
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </form>

                {/* Accessible live region for voice feedback */}
                <div className="sr-only" aria-live="polite">
                  {isListening ? 'Microphone is listening. Speak your question now.' : ''}
                </div>
              </section>
            )}

            {/* Empty placeholder if no image has been processed yet */}
            {!captionResult && !ocrResult && !isCameraActive && (
              <div className="bg-[#0c0c0c] border border-white/15 border-dashed p-12 text-center text-zinc-500 space-y-3 reticle-bracket">
                <Layers className="w-10 h-10 mx-auto text-[#D4FF00]/50" />
                <h3 className="text-xs font-mono font-bold uppercase tracking-widest text-zinc-300">
                  AWAITING IMAGE INPUT
                </h3>
                <p className="text-xs max-w-md mx-auto text-zinc-400 font-sans leading-relaxed">
                  Open your camera or select an image on the left, then click <strong>&ldquo;DESCRIBE SCENE &amp; EXTRACT TEXT&rdquo;</strong> to initialize multimodal captioning, text extraction, speech synthesis, and grounded VQA.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      <AppFooter />
    </div>
  );
};

export default AccessibilityApp;
