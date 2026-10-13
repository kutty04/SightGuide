import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface NeuralConstellationProps {
  className?: string;
}

/**
 * Creates a soft radial glowing circle texture using an offscreen canvas.
 * This guarantees smooth circular particles across all GPUs without external image assets.
 */
function createParticleTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.3, 'rgba(255, 255, 255, 0.8)');
    gradient.addColorStop(0.7, 'rgba(255, 255, 255, 0.2)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 32, 32);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function isWebGLAvailable(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch {
    return false;
  }
}

/**
 * SightGuide Atmospheric Neural Constellation.
 * Decorative 3D particle space rendered via Three.js.
 * Explicitly non-scientific/decorative ambient visualization.
 */
export const NeuralConstellation: React.FC<NeuralConstellationProps> = ({ className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [webglSupported] = useState<boolean>(isWebGLAvailable);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !webglSupported) return;

    // 1. Reduced Motion Preference Check
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // 2. Responsive Particle Budget
    const isMobile = window.innerWidth < 768;
    const particleCount = isMobile ? 45 : 130;
    const maxDistance = isMobile ? 55 : 75;

    // 3. WebGL Initialization with Error Boundary Guard
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: !isMobile,
        powerPreference: 'low-power',
      });
    } catch {
      return;
    }


    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.position = 'absolute';
    renderer.domElement.style.top = '0';
    renderer.domElement.style.left = '0';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.pointerEvents = 'none';

    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 1000);
    camera.position.z = 240;

    // 4. Color Palette (Cyan #38bdf8, Deep Blue #3b82f6, Amethyst #a855f7)
    const colorCyan = new THREE.Color(0x38bdf8);
    const colorBlue = new THREE.Color(0x3b82f6);
    const colorAmethyst = new THREE.Color(0xa855f7);

    // 5. Generate Particle Geometries & Buffers
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const velocities: { x: number; y: number; z: number }[] = [];

    const fieldSpreadX = isMobile ? 220 : 340;
    const fieldSpreadY = isMobile ? 180 : 260;
    const fieldSpreadZ = 120;

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * fieldSpreadX;
      positions[i3 + 1] = (Math.random() - 0.5) * fieldSpreadY;
      positions[i3 + 2] = (Math.random() - 0.5) * fieldSpreadZ;

      // Restrained palette distribution
      const randColor = Math.random();
      const chosenColor = randColor < 0.55 ? colorCyan : randColor < 0.85 ? colorBlue : colorAmethyst;
      colors[i3] = chosenColor.r;
      colors[i3 + 1] = chosenColor.g;
      colors[i3 + 2] = chosenColor.b;

      velocities.push({
        x: (Math.random() - 0.5) * 0.18,
        y: (Math.random() - 0.5) * 0.18,
        z: (Math.random() - 0.5) * 0.1,
      });
    }

    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleTexture = createParticleTexture();
    const pointsMaterial = new THREE.PointsMaterial({
      size: isMobile ? 3.5 : 4.5,
      map: particleTexture,
      transparent: true,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.85,
    });

    const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointCloud);

    // 6. Dynamic Connecting Filaments Buffer
    const maxLineSegments = particleCount * 6;
    const linePositions = new Float32Array(maxLineSegments * 6);
    const lineColors = new Float32Array(maxLineSegments * 6);

    const linesGeometry = new THREE.BufferGeometry();
    linesGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    linesGeometry.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));

    const linesMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.35,
    });

    const lineSegments = new THREE.LineSegments(linesGeometry, linesMaterial);
    scene.add(lineSegments);

    // 7. Interactive Cursor Tracking (Smooth, Subtle Lerp)
    let targetMouseX = 0;
    let targetMouseY = 0;
    let currentMouseX = 0;
    let currentMouseY = 0;

    const handlePointerMove = (e: PointerEvent) => {
      targetMouseX = (e.clientX / window.innerWidth) * 2 - 1;
      targetMouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });

    // 8. Resize Handler
    const handleResize = () => {
      if (!container) return;
      const newW = container.clientWidth || window.innerWidth;
      const newH = container.clientHeight || window.innerHeight;
      camera.aspect = newW / newH;
      camera.updateProjectionMatrix();
      renderer.setSize(newW, newH);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    };

    window.addEventListener('resize', handleResize, { passive: true });

    // 9. Tab Visibility Lifecycle Management
    let isTabVisible = !document.hidden;
    const handleVisibilityChange = () => {
      isTabVisible = !document.hidden;
      if (isTabVisible && !prefersReducedMotion) {
        lastTime = performance.now();
        animate();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    // 10. WebGL Context Loss Handlers
    const handleContextLost = (event: Event) => {
      event.preventDefault();
      cancelAnimationFrame(animationFrameId);
    };

    const handleContextRestored = () => {
      handleResize();
      if (!prefersReducedMotion) animate();
    };

    const canvasElement = renderer.domElement;
    canvasElement.addEventListener('webglcontextlost', handleContextLost, false);
    canvasElement.addEventListener('webglcontextrestored', handleContextRestored, false);

    // 11. Render Loop
    let animationFrameId: number;
    let lastTime = performance.now();

    const animate = () => {
      if (!isTabVisible) return;

      const now = performance.now();
      const delta = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      // Smooth pointer interpolation
      currentMouseX += (targetMouseX - currentMouseX) * 0.04;
      currentMouseY += (targetMouseY - currentMouseY) * 0.04;

      camera.position.x = currentMouseX * 18;
      camera.position.y = currentMouseY * 12;
      camera.lookAt(scene.position);

      const posArray = pointsGeometry.attributes.position.array as Float32Array;

      // Update particle positions
      const halfX = fieldSpreadX / 2;
      const halfY = fieldSpreadY / 2;
      const halfZ = fieldSpreadZ / 2;

      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        posArray[i3] += velocities[i].x * 60 * delta;
        posArray[i3 + 1] += velocities[i].y * 60 * delta;
        posArray[i3 + 2] += velocities[i].z * 60 * delta;

        // Soft bounce boundaries
        if (posArray[i3] < -halfX || posArray[i3] > halfX) velocities[i].x *= -1;
        if (posArray[i3 + 1] < -halfY || posArray[i3 + 1] > halfY) velocities[i].y *= -1;
        if (posArray[i3 + 2] < -halfZ || posArray[i3 + 2] > halfZ) velocities[i].z *= -1;
      }
      pointsGeometry.attributes.position.needsUpdate = true;

      // Update connecting filament lines
      let lineIndex = 0;
      const linePosArray = linesGeometry.attributes.position.array as Float32Array;
      const lineColArray = linesGeometry.attributes.color.array as Float32Array;

      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        const x1 = posArray[i3];
        const y1 = posArray[i3 + 1];
        const z1 = posArray[i3 + 2];

        for (let j = i + 1; j < particleCount; j++) {
          const j3 = j * 3;
          const dx = x1 - posArray[j3];
          const dy = y1 - posArray[j3 + 1];
          const dz = z1 - posArray[j3 + 2];
          const distSq = dx * dx + dy * dy + dz * dz;

          if (distSq < maxDistance * maxDistance && lineIndex < maxLineSegments * 6) {
            const alpha = 1.0 - Math.sqrt(distSq) / maxDistance;

            // Vertex 1
            linePosArray[lineIndex] = x1;
            linePosArray[lineIndex + 1] = y1;
            linePosArray[lineIndex + 2] = z1;
            lineColArray[lineIndex] = colors[i3] * alpha;
            lineColArray[lineIndex + 1] = colors[i3 + 1] * alpha;
            lineColArray[lineIndex + 2] = colors[i3 + 2] * alpha;

            // Vertex 2
            linePosArray[lineIndex + 3] = posArray[j3];
            linePosArray[lineIndex + 4] = posArray[j3 + 1];
            linePosArray[lineIndex + 5] = posArray[j3 + 2];
            lineColArray[lineIndex + 3] = colors[j3] * alpha;
            lineColArray[lineIndex + 4] = colors[j3 + 1] * alpha;
            lineColArray[lineIndex + 5] = colors[j3 + 2] * alpha;

            lineIndex += 6;
          }
        }
      }

      linesGeometry.setDrawRange(0, lineIndex / 3);
      linesGeometry.attributes.position.needsUpdate = true;
      linesGeometry.attributes.color.needsUpdate = true;

      renderer.render(scene, camera);

      if (!prefersReducedMotion) {
        animationFrameId = requestAnimationFrame(animate);
      }
    };

    if (prefersReducedMotion) {
      // Single frame render for reduced motion preference
      renderer.render(scene, camera);
    } else {
      animate();
    }

    // 12. StrictMode Safe Cleanup Handler
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('resize', handleResize);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      canvasElement.removeEventListener('webglcontextlost', handleContextLost);
      canvasElement.removeEventListener('webglcontextrestored', handleContextRestored);

      pointsGeometry.dispose();
      pointsMaterial.dispose();
      particleTexture.dispose();
      linesGeometry.dispose();
      linesMaterial.dispose();

      renderer.dispose();
      if (canvasElement.parentElement === container) {
        container.removeChild(canvasElement);
      }
    };
  }, [webglSupported]);


  if (!webglSupported) {
    // Graceful CSS fallback when WebGL is unavailable
    return (
      <div
        className={`pointer-events-none fixed inset-0 z-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-950/20 via-transparent to-transparent ${className}`}
        aria-hidden="true"
      />
    );
  }

  return (
    <div
      ref={containerRef}
      className={`pointer-events-none fixed inset-0 z-0 overflow-hidden ${className}`}
      aria-hidden="true"
    />
  );
};
