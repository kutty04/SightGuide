import React from 'react';
import { AppNavbar } from '../components/layout/AppNavbar';
import { AppFooter } from '../components/layout/AppFooter';
import { Link } from '../router';
import {
  Smartphone,
  Globe2,
  MessageSquareText,
  Users,
  Video,
  ArrowRight,
} from 'lucide-react';

export const FutureScope: React.FC = () => {
  const roadmapItems = [
    {
      id: '01',
      title: 'Mobile On-Device Edge Deployment',
      subtitle: 'React Native / Flutter with INT8 Quantized Vision Models',
      icon: Smartphone,
      description:
        'Porting the SightGuide pipeline to mobile devices using ONNX Runtime Mobile or TFLite. Integrating INT8 quantized vision-language backbones to enable zero-latency, private, and fully offline assistance on smartphones without cellular connectivity.',
      readiness: 'PLANNED NEXT PHASE',
      technicalFocus: 'ONNX Runtime Mobile &bull; INT8 Quantization &bull; CameraX Streaming',
    },
    {
      id: '02',
      title: 'Multilingual OCR & Global Speech Voices',
      subtitle: 'RapidOCR Multi-Dictionary & Polyglot TTS Synthesis',
      icon: Globe2,
      description:
        'Expanding RapidOCR with language-specific character dictionaries across 10+ languages (Spanish, Hindi, German, French, Japanese, Mandarin) combined with localized native TTS voices for diverse blind and low-vision communities worldwide.',
      readiness: 'ARCHITECTURE READY',
      technicalFocus: 'RapidOCR multi-language models &bull; Polyglot SpeechSynthesis &bull; Locale Detection',
    },
    {
      id: '03',
      title: 'Interactive Conversational VQA (Visual Question Answering)',
      subtitle: 'Multi-Turn Contextual Dialog for Scene Clarification',
      icon: MessageSquareText,
      description:
        'Allowing users to ask follow-up questions about the scene (e.g. "What color is the pedestrian light?", "What does the expiration date on the bottle say?"). The LLM maintains dialogue memory conditioned on image tokens and OCR boxes.',
      readiness: 'PROTOTYPE SPEC',
      technicalFocus: 'Conversation Memory &bull; Region-of-Interest Cropping &bull; Grounded VQA',
    },
    {
      id: '04',
      title: 'Human-Centered Accessibility Study & Reference Annotations',
      subtitle: 'Formal Blind & Low-Vision Usability Protocols',
      icon: Users,
      description:
        'Conducting an empirical human-subject study with visually impaired users to measure cognitive load, caption comprehension, and auditory clarity. Crowdsourcing 5 human reference captions per image across the 96-image dataset for formal CIDEr, SPICE, and METEOR benchmarking.',
      readiness: 'PROTOCOL DRAFTED',
      technicalFocus: 'IRB Protocol &bull; 5x Human Annotations/image &bull; CIDEr / SPICE / BLEU-4 Metrics',
    },
    {
      id: '05',
      title: 'Wearable Smart Glasses & Real-Time Video Streaming',
      subtitle: 'Sliding-Window Keyframe Sampling for Wearables',
      icon: Video,
      description:
        'Interfacing with smart glasses hardware (e.g., Raspberry Pi camera rig or Bluetooth audio glasses) to process 1 frame every 2 seconds, delivering continuous ambient audio descriptions of hazards, doors, and oncoming objects.',
      readiness: 'RESEARCH CONCEPT',
      technicalFocus: 'Keyframe Selection &bull; Low-Power DSP &bull; BLE Audio Streaming',
    },
  ];

  return (
    <div className="min-h-screen bg-[#050505] text-[#F7F6F2] flex flex-col font-sans selection:bg-[#D4FF00] selection:text-black">
      <AppNavbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-widest text-[#D4FF00] font-bold bg-[#D4FF00]/10 border border-[#D4FF00]/40 px-2.5 py-0.5">
              06 // RESEARCH &amp; ENGINEERING ROADMAP
            </span>
            <span className="text-[10px] font-mono text-[#D4FF00] bg-black border border-white/20 px-2 py-0.5">
              POST-GRADUATION SCOPE
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight mt-1.5">
            FUTURE SCOPE &amp; SYSTEM EVOLUTION DIRECTIONS
          </h1>
          <p className="text-sm text-zinc-400 max-w-3xl leading-relaxed font-sans">
            SightGuide V2 delivers a robust CPU-optimized foundation combining diversity MMR captioning, RapidOCR text extraction, and Web Speech API playback. Below is our realistic future roadmap for mobile translation, conversational VQA, and human-subject evaluation.
          </p>
        </div>

        {/* Roadmap Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {roadmapItems.map((item) => {
            const Icon = item.icon;
            return (
              <div
                key={item.id}
                className="bg-[#0c0c0c] border border-white/10 p-6 space-y-4 shadow-xl flex flex-col justify-between group hover:border-[#D4FF00]/60 transition-all reticle-bracket"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 border border-white/20 bg-black flex items-center justify-center text-[#D4FF00] group-hover:bg-[#D4FF00] group-hover:text-black transition-colors">
                        <Icon className="w-4 h-4" />
                      </div>
                      <span className="font-display font-bold text-3xl text-zinc-700 group-hover:text-[#D4FF00] transition-colors">
                        {item.id}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono uppercase bg-black text-[#D4FF00] border border-[#D4FF00]/40 px-2 py-0.5">
                        {item.readiness}
                      </span>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-base font-bold text-white tracking-tight font-sans">
                      {item.title}
                    </h3>
                    <p className="text-xs text-[#D4FF00] font-mono mt-0.5">
                      {item.subtitle}
                    </p>
                  </div>

                  <p className="text-xs text-zinc-300 leading-relaxed font-sans">
                    {item.description}
                  </p>
                </div>

                <div className="pt-3 border-t border-white/10 text-[11px] font-mono text-zinc-400">
                  <strong className="text-[#D4FF00]">STACK: </strong>
                  <span dangerouslySetInnerHTML={{ __html: item.technicalFocus }} />
                </div>
              </div>
            );
          })}
        </div>

        {/* Call to Action Bar */}
        <div className="bg-[#0c0c0c] border border-white/10 p-6 flex flex-col sm:flex-row items-center justify-between gap-4 reticle-bracket">
          <div className="space-y-1 text-center sm:text-left">
            <h3 className="text-sm font-mono font-bold uppercase tracking-wider text-white">
              EXPERIENCE THE LIVE V2 ACCESSIBILITY ASSISTANT NOW
            </h3>
            <p className="text-xs text-zinc-400 font-sans">
              Upload any photograph to test diversity MMR captioning, RapidOCR, and text-to-speech.
            </p>
          </div>
          <Link
            to="/app"
            className="inline-flex items-center gap-2 px-5 py-3 bg-[#D4FF00] hover:bg-[#ccff00] text-black font-mono font-bold text-xs uppercase tracking-wider transition shadow-[0_0_15px_rgba(212,255,0,0.25)]"
          >
            <span>LAUNCH ACCESSIBILITY APP</span>
            <ArrowRight className="w-4 h-4 text-black" />
          </Link>
        </div>
      </main>

      <AppFooter />
    </div>
  );
};

export default FutureScope;
