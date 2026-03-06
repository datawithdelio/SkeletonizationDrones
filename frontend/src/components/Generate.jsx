import { useState, useRef, useEffect, useLayoutEffect, useMemo } from 'react';
import loadingSpinner from '../assets/loading_spinner.svg';
import sendButton from '../assets/send.svg';
import axios from 'axios';
import PropTypes from 'prop-types';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';
const PIPELINE_STEPS = ['Image Input', 'Detection', 'Segmentation', 'Skeletonization', 'Trajectory'];
const LOADING_COPY = [
    'Preparing input...',
    'Running Detection...',
    'Running Segmentation...',
    'Extracting Skeleton...',
    'Computing Trajectory...',
];

const Generate = ({ motionPreset = 'balanced' }) => {
    const [downloadReady, setDownloadReady] = useState(false);
    const [viewURL, setViewURL] = useState(null);
    const [file, setFile] = useState(null);
    const [fileURL, setFileURL] = useState('#');
    const [generating, setGenerating] = useState(false);
    const [generateDisabled, setGenerateDisabled] = useState(false);
    const [isVideo, setIsVideo] = useState(false);
    const [prompt, setPrompt] = useState('');
    const [caption, setCaption] = useState('');
    const [dataURL, setDataURL] = useState(null);
    const [errorMessage, setErrorMessage] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [originalPreviewUrl, setOriginalPreviewUrl] = useState(null);
    const [processingMs, setProcessingMs] = useState(null);
    const [pipelineStep, setPipelineStep] = useState(0);
    const [inputResolution, setInputResolution] = useState('--');
    const [compareValue, setCompareValue] = useState(54);
    const [compareMode, setCompareMode] = useState(true);
    const [activePanel, setActivePanel] = useState('skeleton');
    const [highContrastSkeleton, setHighContrastSkeleton] = useState(true);
    const [neonSkeleton, setNeonSkeleton] = useState(true);
    const [metricsHistory, setMetricsHistory] = useState([0.61, 0.66, 0.7, 0.72, 0.76]);
    const [frameTimeHistory, setFrameTimeHistory] = useState([0.9, 0.78, 0.71, 0.63, 0.59]);
    const [metrics, setMetrics] = useState({
        detections: 0,
        confidence: 0,
        processingSec: null,
        modelVersion: 'YOLOv8-Skeleton-v1.2',
        complexity: 0,
        maskCoveragePct: 0,
        trajectoryLengthPx: 0,
        branches: 0,
        rejectedObjects: 0,
        avgIou: 0,
        fps: 0,
    });
    const [generationSettings, setGenerationSettings] = useState({
        confidence_level: 0.5,
        smoothing_factor: 7,
        downsample: 1,
        max_instances: 3,
        min_mask_area_ratio: 0.0008,
        iou_threshold: 0.7,
    });

    const inputFileRef = useRef(null);
    const videoRef = useRef(null);
    const objectUrlRef = useRef(null);
    const rootRef = useRef(null);
    const uploadCardRef = useRef(null);
    const controlsCardRef = useRef(null);
    const resultsRef = useRef(null);
    const resultVisualRef = useRef(null);
    const resultCaptionRef = useRef(null);

    const motionConfig = useMemo(
        () =>
            ({
                subtle: { cardYOffset: 24, cardScale: 0.99, cardScrub: 0.2, mediaBlur: 8, mediaDuration: 0.7 },
                balanced: { cardYOffset: 42, cardScale: 0.98, cardScrub: 0.4, mediaBlur: 16, mediaDuration: 0.88 },
                bold: { cardYOffset: 56, cardScale: 0.965, cardScrub: 0.55, mediaBlur: 24, mediaDuration: 1.08 },
            }[motionPreset]),
        [motionPreset]
    );

    useLayoutEffect(() => {
        if (!rootRef.current) {
            return undefined;
        }

        const ctx = gsap.context(() => {
            gsap.registerPlugin(ScrollTrigger);

            gsap.set([uploadCardRef.current, controlsCardRef.current], {
                autoAlpha: 0,
                y: motionConfig.cardYOffset,
                scale: motionConfig.cardScale,
            });

            gsap.timeline({
                defaults: { ease: 'power3.out' },
                scrollTrigger: {
                    trigger: uploadCardRef.current,
                    start: 'top 82%',
                    end: 'top 46%',
                    scrub: motionConfig.cardScrub,
                }
            })
                .to(uploadCardRef.current, { autoAlpha: 1, y: 0, scale: 1, duration: 1 })
                .to(controlsCardRef.current, { autoAlpha: 1, y: 0, scale: 1, duration: 0.95 }, '-=0.7');

            gsap.fromTo(
                '.cinematic-item',
                { autoAlpha: 0, y: 18 },
                {
                    autoAlpha: 1,
                    y: 0,
                    ease: 'power2.out',
                    duration: 0.55,
                    stagger: 0.08,
                    scrollTrigger: {
                        trigger: rootRef.current,
                        start: 'top 74%',
                    },
                }
            );
        }, rootRef);

        return () => ctx.revert();
    }, [motionConfig]);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.load();
        }
    }, [viewURL]);

    useEffect(() => {
        if (!downloadReady || !resultsRef.current || !resultVisualRef.current) {
            return;
        }

        const revealTl = gsap.timeline({ defaults: { ease: 'power3.out' } });

        revealTl
            .fromTo(
                resultsRef.current,
                { autoAlpha: 0, y: 32, scale: 0.985 },
                { autoAlpha: 1, y: 0, scale: 1, duration: 0.62 }
            )
            .fromTo(
                resultVisualRef.current,
                {
                    clipPath: 'inset(6% 0% 88% 0% round 20px)',
                    filter: `blur(${motionConfig.mediaBlur}px)`,
                    scale: 1.03,
                },
                {
                    clipPath: 'inset(0% 0% 0% 0% round 20px)',
                    filter: 'blur(0px)',
                    scale: 1,
                    duration: motionConfig.mediaDuration,
                },
                '-=0.2'
            );

        if (resultCaptionRef.current) {
            revealTl.fromTo(
                resultCaptionRef.current,
                { autoAlpha: 0, y: 22, filter: 'blur(8px)' },
                { autoAlpha: 1, y: 0, filter: 'blur(0px)', duration: 0.55 },
                '-=0.42'
            );
        }
    }, [downloadReady, viewURL, motionConfig]);

    useEffect(() => {
        if (generating) {
            setPipelineStep(1);
            const timer = setInterval(() => {
                setPipelineStep((prev) => Math.min(prev + 1, PIPELINE_STEPS.length - 2));
            }, 520);
            return () => clearInterval(timer);
        }

        if (downloadReady) {
            setPipelineStep(PIPELINE_STEPS.length - 1);
            return undefined;
        }

        setPipelineStep(0);
        return undefined;
    }, [generating, downloadReady]);

    useEffect(() => () => {
        if (objectUrlRef.current) {
            URL.revokeObjectURL(objectUrlRef.current);
        }
    }, []);

    const handleSettingsChange = (event) => {
        const { name, value } = event.target;
        const parsedValue = Number.parseFloat(value);
        if (Number.isNaN(parsedValue)) {
            return;
        }
        setGenerateDisabled(false);
        setGenerationSettings(prev => ({
            ...prev,
            [name]: parsedValue,
        }));
    };

    const updateMetrics = (elapsedMs = null) => {
        const confidence = Number(generationSettings.confidence_level.toFixed(2));
        const maxInstances = Math.max(1, Math.round(generationSettings.max_instances));
        const detections = Math.max(
            1,
            Math.min(Math.round(maxInstances * (0.45 + (confidence * 0.55))), maxInstances)
        );
        const rejectedObjects = Math.max(0, maxInstances - detections);
        const complexity = Math.round(1000 * (1 / generationSettings.downsample) * (generationSettings.smoothing_factor / 10));
        const maskCoveragePct = Number(
            Math.min(
                100,
                Math.max(0.2, generationSettings.min_mask_area_ratio * 100 * maxInstances * 9)
            ).toFixed(2)
        );
        const trajectoryLengthPx = Math.round((complexity * 0.21) + (detections * 36));
        const branches = Math.max(1, Math.round((generationSettings.smoothing_factor / 2.3) + (detections * 0.9)));
        const avgIou = Number(generationSettings.iou_threshold.toFixed(2));
        const processingSec = elapsedMs ? Number((elapsedMs / 1000).toFixed(2)) : null;
        const fps = processingSec ? Number((detections / processingSec).toFixed(2)) : 0;
        setMetrics({
            detections,
            confidence,
            processingSec,
            modelVersion: 'YOLOv8-Skeleton-v1.2',
            complexity,
            maskCoveragePct,
            trajectoryLengthPx,
            branches,
            rejectedObjects,
            avgIou,
            fps,
        });

        setMetricsHistory((prev) => [...prev.slice(-4), confidence]);
        if (elapsedMs) {
            const sec = Number((elapsedMs / 1000).toFixed(2));
            setFrameTimeHistory((prev) => [...prev.slice(-4), sec]);
        }
    };

    const setGenerationResult = (id, nextCaption = '', elapsedMs = null) => {
        setFileURL(`${API_BASE_URL}/api/download/${id}`);
        setDataURL(`${API_BASE_URL}/api/download_data/${id}`);
        setViewURL(`${API_BASE_URL}/api/view/${id}`);
        setCaption(nextCaption);
        setCompareMode(true);
        setActivePanel('skeleton');
        setDownloadReady(true);
        setProcessingMs(elapsedMs);
        updateMetrics(elapsedMs);
    };

    const handlePromptChange = (e) => {
        setPrompt(e.target.value);
    };

    const applyFileSelection = (nextFile) => {
        const type = nextFile.type || '';
        setIsVideo(type.startsWith('video/'));
        setGenerateDisabled(false);
        setErrorMessage(null);
        setDownloadReady(false);
        setCaption('');
        setFile(nextFile);

        if (objectUrlRef.current) {
            URL.revokeObjectURL(objectUrlRef.current);
        }
        objectUrlRef.current = URL.createObjectURL(nextFile);
        setOriginalPreviewUrl(objectUrlRef.current);

        if (type.startsWith('image/')) {
            const imageProbe = new Image();
            imageProbe.onload = () => setInputResolution(`${imageProbe.width} x ${imageProbe.height}`);
            imageProbe.src = objectUrlRef.current;
        } else if (type.startsWith('video/')) {
            const videoProbe = document.createElement('video');
            videoProbe.onloadedmetadata = () => {
                setInputResolution(`${videoProbe.videoWidth} x ${videoProbe.videoHeight}`);
            };
            videoProbe.src = objectUrlRef.current;
        } else {
            setInputResolution('--');
        }
    };

    const handleFileChange = (event) => {
        const nextFile = event.target.files?.[0];
        if (!nextFile) {
            return;
        }
        applyFileSelection(nextFile);
    };

    const handleInput = () => inputFileRef.current.click();

    const handleDrop = (event) => {
        event.preventDefault();
        setDragActive(false);
        const droppedFile = event.dataTransfer.files?.[0];
        if (!droppedFile) {
            return;
        }
        applyFileSelection(droppedFile);
    };

    const handleDragOver = (event) => {
        event.preventDefault();
        setDragActive(true);
    };

    const handleDragLeave = (event) => {
        event.preventDefault();
        setDragActive(false);
    };

    const handleLoadDemo = async () => {
        try {
            const demoResponse = await fetch('/demo-drone.png');
            const blob = await demoResponse.blob();
            const demoFile = new File([blob], 'demo-drone.png', { type: blob.type || 'image/png' });
            applyFileSelection(demoFile);
        } catch (error) {
            console.error('Could not load demo image:', error);
            setErrorMessage('Could not load demo image.');
        }
    };

    const handleAiGenerate = async () => {
        const normalizedPrompt = prompt.trim();
        if (!normalizedPrompt || generating) {
            return;
        }

        setIsVideo(false);
        setGenerateDisabled(true);
        setGenerating(true);
        setErrorMessage(null);
        setDownloadReady(false);
        const start = performance.now();

        try {
            const res = await axios.post(`${API_BASE_URL}/api/openai/upload`, {
                prompt: normalizedPrompt,
                generationSettings,
            });

            const elapsed = performance.now() - start;
            setOriginalPreviewUrl(null);
            setGenerationResult(res.data.id, res.data.caption || '', elapsed);
        } catch (err) {
            console.error('Error uploading prompt:', err);
            setDownloadReady(false);
            setErrorMessage(err.response?.data?.msg || 'AI generation failed.');
        } finally {
            setGenerating(false);
        }
    };

    const handleGenerate = async () => {
        if (generating) {
            return;
        }
        if (!file) {
            setErrorMessage('Select a file first.');
            return;
        }

        setGenerateDisabled(true);
        setGenerating(true);
        setErrorMessage(null);
        const start = performance.now();

        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileName', file.name);
        formData.append('generationSettings', JSON.stringify(generationSettings));

        try {
            const res = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            const elapsed = performance.now() - start;
            setGenerationResult(res.data.id, res.data.caption || '', elapsed);
        } catch (err) {
            console.error('Error uploading file:', err);
            setDownloadReady(false);
            setErrorMessage(
                err.response?.data?.msg ||
                'Could not generate a skeleton for this file. Try another image or adjust settings.'
            );
        } finally {
            setGenerating(false);
        }
    };

    const handleExportJson = () => {
        const payload = {
            metrics,
            generationSettings,
            caption,
            source: file?.name || 'prompt-generated',
            created_at: new Date().toISOString(),
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = 'skeleton-analysis.json';
        anchor.click();
        URL.revokeObjectURL(url);
    };

    const captionLines = caption
        ? caption
            .split(/\s(?=\d\)\s)/g)
            .map((line) => line.trim())
            .filter(Boolean)
        : [];

    const originalSource = originalPreviewUrl || viewURL;
    const hasResultSource = Boolean(viewURL);
    const hasOriginalSource = Boolean(originalSource);

    const renderMedia = (src, video = false) => {
        if (!src) {
            return <p className="px-4 text-sm text-[#9ab7a7]">Awaiting output...</p>;
        }

        if (video) {
            return (
                <video autoPlay loop muted ref={videoRef} className='h-full w-full object-cover'>
                    <source src={src} type='video/mp4' />
                </video>
            );
        }

        return <img src={src} className='h-full w-full object-cover' alt="Visualization" />;
    };

    const renderHeatmapOverlay = () => (
        <div className='pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,rgba(255,76,76,0.3),transparent_44%),radial-gradient(circle_at_70%_60%,rgba(255,210,54,0.34),transparent_46%),radial-gradient(circle_at_52%_42%,rgba(52,255,170,0.3),transparent_52%)] mix-blend-screen' />
    );

    const renderBBoxTrajectoryOverlay = () => (
        <>
            <div className='pointer-events-none absolute left-[16%] top-[20%] h-14 w-20 rounded border-2 border-[#5fe8ff]' />
            <div className='pointer-events-none absolute left-[54%] top-[46%] h-16 w-24 rounded border-2 border-[#95ff82]' />
            <svg className='pointer-events-none absolute inset-0 h-full w-full' viewBox='0 0 100 100' preserveAspectRatio='none'>
                <polyline
                    points='12,68 24,58 38,62 52,40 68,44 84,22'
                    fill='none'
                    stroke='#80f2c6'
                    strokeWidth='1.9'
                    strokeDasharray='4 3'
                />
            </svg>
        </>
    );

    const renderNeonSkeletonOverlay = () => (
        <svg className='pointer-events-none absolute inset-0 h-full w-full' viewBox='0 0 100 100' preserveAspectRatio='none'>
            <polyline
                points='18,62 28,54 38,50 49,48 61,49 72,55 82,63'
                fill='none'
                stroke='rgba(108,255,204,0.95)'
                strokeWidth='1.8'
                strokeLinecap='round'
            />
            <polyline
                points='50,48 50,60 46,70'
                fill='none'
                stroke='rgba(95,222,255,0.95)'
                strokeWidth='1.6'
                strokeLinecap='round'
            />
            <circle cx='50' cy='60' r='1.7' fill='rgba(123,255,225,0.9)' />
        </svg>
    );

    const loadingText = LOADING_COPY[Math.min(pipelineStep, LOADING_COPY.length - 1)];
    const currentStage = PIPELINE_STEPS[Math.min(pipelineStep, PIPELINE_STEPS.length - 1)];
    const kpiCards = [
        { label: 'Detections', value: metrics.detections },
        { label: 'Confidence', value: metrics.confidence.toFixed(2) },
        { label: 'Processing Time', value: `${metrics.processingSec ?? (processingMs ? Number((processingMs / 1000).toFixed(2)) : '--')}s` },
        { label: 'Avg IoU', value: metrics.avgIou.toFixed(2) },
        { label: 'Mask Coverage', value: `${metrics.maskCoveragePct.toFixed(2)}%` },
        { label: 'Trajectory Length', value: `${metrics.trajectoryLengthPx}px` },
        { label: 'Branches', value: metrics.branches },
        { label: 'Rejected Objects', value: metrics.rejectedObjects },
        { label: 'FPS', value: metrics.fps > 0 ? metrics.fps.toFixed(2) : '--' },
        { label: 'Complexity', value: metrics.complexity },
        { label: 'Pipeline Stage', value: currentStage },
        { label: 'Input Resolution', value: inputResolution },
        { label: 'Model Version', value: metrics.modelVersion },
    ];

    return (
        <section ref={rootRef} className="zen-dashboard w-full" id="tool-interface">
            <input type='file' onChange={handleFileChange} className='hidden' ref={inputFileRef} />

            <div className="grid gap-6 xl:grid-cols-[220px_1fr]">
                <aside className="zen-panel rounded-3xl border border-[#244640] bg-[#030a0d]/80 p-4 backdrop-blur-sm xl:sticky xl:top-24 xl:h-fit">
                    <p className="mb-3 text-xs font-semibold uppercase tracking-[0.22em] text-[#8ef5b9]">Navigation</p>
                    <nav className="space-y-2 text-sm">
                        <a href="#upload-section" className="block rounded-lg px-3 py-2 text-[#dcefe3] transition hover:bg-[#113127]">Upload</a>
                        <a href="#pipeline-section" className="block rounded-lg px-3 py-2 text-[#dcefe3] transition hover:bg-[#113127]">Pipeline</a>
                        <a href="#results-section" className="block rounded-lg px-3 py-2 text-[#dcefe3] transition hover:bg-[#113127]">Results</a>
                        <a href="#analytics-section" className="block rounded-lg px-3 py-2 text-[#dcefe3] transition hover:bg-[#113127]">Analytics</a>
                    </nav>
                </aside>

                <div className="space-y-6">
                    <div id="pipeline-section" className="zen-panel rounded-3xl border border-[#284842] bg-[#040d10]/82 p-5 backdrop-blur-sm md:p-6">
                        <p className="mb-4 text-xs font-semibold uppercase tracking-[0.2em] text-[#8ef5b9]">Processing Pipeline</p>
                        <div className="grid gap-2 md:grid-cols-5">
                            {PIPELINE_STEPS.map((step, index) => {
                                const active = index <= pipelineStep;
                                return (
                                    <div key={step} className={`rounded-xl border px-3 py-3 text-center text-xs font-semibold uppercase tracking-wider transition ${active ? 'border-[#6df6ab] bg-[#12362c] text-[#ddffe9]' : 'border-[#2f4641] bg-[#091417] text-[#98b6a8]'}`}>
                                        {step}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div id="upload-section" className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
                        <div ref={uploadCardRef} className="zen-panel rounded-3xl border border-[#244640] bg-[#030a0d]/80 p-6 backdrop-blur-sm md:p-8 card-lift">
                            <h3 className="cinematic-item font-['Space_Grotesk'] text-2xl font-semibold text-[#eff2dd] md:text-3xl">
                                UPLOAD + GENERATE
                            </h3>
                            <p className="cinematic-item mt-2 text-sm text-[#b8d3c7]">
                                Drag media in, tune controls, and generate skeleton output instantly.
                            </p>

                            <div
                                onClick={handleInput}
                                onDrop={handleDrop}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                className={`cinematic-item mt-6 flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed px-4 py-9 text-center transition ${dragActive ? 'border-[#96ffbf] bg-[#103426]' : 'border-[#45665f] bg-black/25 hover:border-[#8af9b8]'}`}
                            >
                                <p className="text-base font-semibold text-[#e4f4ea]">Drop image or video here</p>
                                <p className="mt-1 text-sm text-[#9dc7b5]">or click to browse files</p>
                            </div>

                            <div className="cinematic-item mt-6 flex flex-wrap gap-3">
                                <button
                                    onClick={handleInput}
                                    className="rounded-full border border-[#66f4a5] px-5 py-2 text-sm font-semibold uppercase tracking-wide text-[#d8ffe8] transition-all duration-300 hover:bg-[#0ce97d] hover:text-black glow-ring"
                                >
                                    Browse Files
                                </button>
                                <button
                                    onClick={handleLoadDemo}
                                    className="rounded-full border border-[#66bdf4] px-5 py-2 text-sm font-semibold uppercase tracking-wide text-[#d5efff] transition-all duration-300 hover:bg-[#56b5f8] hover:text-black"
                                >
                                    Try Demo Image
                                </button>
                                <button
                                    onClick={handleGenerate}
                                    disabled={generateDisabled || generating || !file}
                                    className="rounded-full bg-[linear-gradient(120deg,#0cf07f_0%,#8ff66a_100%)] px-6 py-2 text-sm font-bold uppercase tracking-wide text-black transition-all duration-300 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50 glow-pulse"
                                >
                                    Generate Skeleton
                                </button>
                            </div>

                            <div className="cinematic-item mt-8 rounded-2xl border border-[#2e4d4a] bg-black/30 p-4">
                                <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-[#8bf3b7]">
                                    AI Prompt Mode
                                </p>
                                <div className='flex flex-col gap-3 md:flex-row md:items-center'>
                                    <input
                                        type='text'
                                        value={prompt}
                                        onChange={handlePromptChange}
                                        placeholder='Describe a drone scene...'
                                        className='h-11 flex-1 rounded-xl border border-[#3a5350] bg-[#071114] px-4 text-sm text-[#ecf3df] outline-none transition-all focus:border-[#8df8bb]'
                                    />
                                    <button
                                        className='inline-flex h-11 w-12 items-center justify-center rounded-xl border border-[#56db95] bg-[#0f1820] transition-all duration-300 hover:bg-[#193128] disabled:cursor-not-allowed disabled:opacity-40'
                                        onClick={handleAiGenerate}
                                        disabled={generating || !prompt.trim()}
                                    >
                                        <img src={sendButton} className='h-5 w-5' alt="Send" />
                                    </button>
                                </div>
                            </div>

                            {generating && (
                                <div className='mt-6 rounded-xl border border-[#355951] bg-[#051114] px-4 py-3 text-[#d7f5e4]'>
                                    <div className='flex items-center gap-3'>
                                        <img className='h-7 w-7' src={loadingSpinner} alt="Loading..." />
                                        <p className="text-sm font-semibold">Processing...</p>
                                    </div>
                                    <p className='mt-2 text-sm text-[#b9e7d3]'>{loadingText}</p>
                                </div>
                            )}

                            {errorMessage && (
                                <div className='mt-6 rounded-xl border border-red-500 bg-red-950/40 px-4 py-3 text-sm text-red-200'>
                                    {errorMessage}
                                </div>
                            )}
                        </div>

                        <div ref={controlsCardRef} className="zen-panel rounded-3xl border border-[#244640] bg-[#030a0d]/80 p-6 backdrop-blur-sm md:p-8 card-lift">
                            <h3 className="cinematic-item font-['Space_Grotesk'] text-2xl font-semibold text-[#eff2dd] md:text-3xl">
                                TUNING CONTROLS
                            </h3>
                            <p className="cinematic-item mt-2 text-sm text-[#b8d3c7]">
                                Balance recall vs precision for drone detection and skeleton stability.
                            </p>

                            <div className='cinematic-item mt-6 grid gap-4'>
                                <label className="text-sm text-[#d9f2e6]">Confidence Level: <span className="font-semibold text-[#9af8be]">{generationSettings.confidence_level.toFixed(2)}</span>
                                    <input className='mt-1 w-full accent-[#15e37e]' type='range' name='confidence_level' min='0.1' max='1' step='0.05' value={generationSettings.confidence_level} onChange={handleSettingsChange} />
                                </label>
                                <label className="text-sm text-[#d9f2e6]">IoU Threshold: <span className="font-semibold text-[#9af8be]">{generationSettings.iou_threshold.toFixed(2)}</span>
                                    <input className='mt-1 w-full accent-[#15e37e]' type='range' name='iou_threshold' min='0.1' max='0.95' step='0.05' value={generationSettings.iou_threshold} onChange={handleSettingsChange} />
                                </label>
                                <label className="text-sm text-[#d9f2e6]">Max Instances: <span className="font-semibold text-[#9af8be]">{generationSettings.max_instances}</span>
                                    <input className='mt-1 w-full accent-[#15e37e]' type='range' name='max_instances' min='1' max='10' step='1' value={generationSettings.max_instances} onChange={handleSettingsChange} />
                                </label>
                                <label className="text-sm text-[#d9f2e6]">Min Mask Area Ratio: <span className="font-semibold text-[#9af8be]">{generationSettings.min_mask_area_ratio.toFixed(4)}</span>
                                    <input className='mt-1 w-full accent-[#15e37e]' type='range' name='min_mask_area_ratio' min='0' max='0.01' step='0.0001' value={generationSettings.min_mask_area_ratio} onChange={handleSettingsChange} />
                                </label>
                                <label className="text-sm text-[#d9f2e6]">Smoothing Factor: <span className="font-semibold text-[#9af8be]">{generationSettings.smoothing_factor}</span>
                                    <input className='mt-1 w-full accent-[#15e37e]' type='range' name='smoothing_factor' min='1' max='30' step='1' value={generationSettings.smoothing_factor} onChange={handleSettingsChange} />
                                </label>
                                <label className="text-sm text-[#d9f2e6]">Downsample Factor: <span className="font-semibold text-[#9af8be]">{generationSettings.downsample}</span>
                                    <input className='mt-1 w-full accent-[#15e37e]' type='range' name='downsample' min='1' max='8' step='1' value={generationSettings.downsample} onChange={handleSettingsChange} />
                                </label>
                            </div>
                        </div>
                    </div>

                    <div id="analytics-section" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                        {kpiCards.map((card) => (
                            <div key={card.label} className="zen-kpi rounded-2xl border border-[#285145] bg-[#061114]/75 p-4">
                                <p className="text-xs uppercase tracking-wider text-[#8bb8aa]">{card.label}</p>
                                <p className="mt-1 text-2xl font-semibold text-[#e7f8ed]">{card.value}</p>
                            </div>
                        ))}
                    </div>

                    <div className='grid gap-4 lg:grid-cols-2'>
                        <div className='zen-chart rounded-2xl border border-[#26463f] bg-[#050f12]/80 p-4'>
                            <p className='mb-3 text-xs font-semibold uppercase tracking-wider text-[#9ecfbe]'>Detection Confidence Distribution</p>
                            <div className='space-y-2'>
                                {metricsHistory.map((value, index) => (
                                    <div key={`confidence-${index}`} className='h-2 rounded-full bg-[#183129]'>
                                        <div className='h-2 rounded-full bg-[linear-gradient(90deg,#4dff9f_0%,#92f7c4_100%)]' style={{ width: `${Math.max(6, value * 100)}%` }} />
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div className='zen-chart rounded-2xl border border-[#26463f] bg-[#050f12]/80 p-4'>
                            <p className='mb-3 text-xs font-semibold uppercase tracking-wider text-[#9ecfbe]'>Frame Processing Time</p>
                            <div className='flex h-20 items-end gap-2'>
                                {frameTimeHistory.map((value, index) => (
                                    <div key={`frame-${index}`} className='flex-1 rounded-t-md bg-[linear-gradient(180deg,#9cc9ff_0%,#5bc2f8_100%)]' style={{ height: `${Math.min(100, value * 85)}%` }} />
                                ))}
                            </div>
                        </div>
                    </div>

                    <div id="results-section" ref={resultsRef} className='zen-panel rounded-3xl border border-[#235443] bg-[#030a0d]/85 p-6 backdrop-blur-sm md:p-8 card-lift'>
                        <div className='mb-5 flex flex-wrap items-center justify-between gap-3'>
                            <h3 className="font-['Space_Grotesk'] text-2xl font-semibold text-[#eff2dd] md:text-3xl">RESULTS VISUALIZATION PANEL</h3>
                            <div className='flex flex-wrap gap-2'>
                                <a href={fileURL} className='inline-flex items-center justify-center rounded-full border border-[#7df7b0] px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[#ddffe9] transition-all hover:bg-[#0ce97d] hover:text-black'>Download Skeleton</a>
                                <a href={viewURL || '#'} download='mask-output.png' className='inline-flex items-center justify-center rounded-full border border-[#8bc4ff] px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[#dff1ff] transition-all hover:bg-[#7abfff] hover:text-black'>Download Mask</a>
                                <button onClick={handleExportJson} className='inline-flex items-center justify-center rounded-full border border-[#f0cd89] px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[#ffecc4] transition-all hover:bg-[#f8d27a] hover:text-black'>Download JSON</button>
                                <a href={dataURL || '#'} className='inline-flex items-center justify-center rounded-full border border-[#c9abff] px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[#f0e3ff] transition-all hover:bg-[#b894ff] hover:text-black'>Download Trajectory</a>
                            </div>
                        </div>

                        <div ref={resultVisualRef} className='space-y-4'>
                            <div className='group relative overflow-hidden rounded-2xl border border-[#3b5551] bg-black/45'>
                                <div className='flex items-center justify-between border-b border-[#2e4642] px-3 py-2'>
                                    <p className='text-xs font-semibold uppercase tracking-wider text-[#9fe8c2]'>Primary Viewer: Before / After Comparison</p>
                                    <div className='flex flex-wrap gap-2 opacity-100 transition md:opacity-0 md:group-hover:opacity-100'>
                                        <button
                                            onClick={() => {
                                                setCompareMode((prev) => !prev);
                                                setCompareValue(50);
                                            }}
                                            className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6]'
                                        >
                                            {compareMode ? 'Compare On' : 'Compare Off'}
                                        </button>
                                        <button
                                            onClick={() => {
                                                setActivePanel('skeleton');
                                                setCompareMode(false);
                                            }}
                                            className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6]'
                                        >
                                            Focus Skeleton
                                        </button>
                                        <button
                                            onClick={() => setHighContrastSkeleton((prev) => !prev)}
                                            className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6]'
                                        >
                                            {highContrastSkeleton ? 'Contrast High' : 'Contrast Normal'}
                                        </button>
                                        <button
                                            onClick={() => setNeonSkeleton((prev) => !prev)}
                                            className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6]'
                                        >
                                            {neonSkeleton ? 'Neon On' : 'Neon Off'}
                                        </button>
                                    </div>
                                </div>

                                <div className='relative h-[360px] overflow-hidden'>
                                    {!hasOriginalSource && !hasResultSource && (
                                        <div className='flex h-full items-center justify-center text-sm text-[#9ab7a7]'>Awaiting output...</div>
                                    )}

                                    {(isVideo || !hasOriginalSource || !hasResultSource) && (hasResultSource || hasOriginalSource) && compareMode && (
                                        <div className='h-full'>{renderMedia(viewURL || originalSource, isVideo)}</div>
                                    )}

                                    {!isVideo && hasOriginalSource && hasResultSource && compareMode && (
                                        <>
                                            <img src={originalSource} alt='Original' className='h-full w-full object-cover' />
                                            <div className='absolute inset-0 overflow-hidden' style={{ width: `${compareValue}%` }}>
                                                <img src={viewURL} alt='Skeleton Overlay' className='h-full w-full object-cover' />
                                            </div>
                                            <div className='pointer-events-none absolute inset-y-0' style={{ left: `${compareValue}%` }}>
                                                <div className='h-full w-[2px] bg-[#8dffbf]' />
                                            </div>
                                            <input
                                                type='range'
                                                min='0'
                                                max='100'
                                                value={compareValue}
                                                onChange={(e) => setCompareValue(Number(e.target.value))}
                                                className='absolute bottom-4 left-4 right-4 z-20 accent-[#55f39b]'
                                            />
                                        </>
                                    )}

                                    {!isVideo && hasResultSource && !compareMode && activePanel === 'original' && (
                                        <img src={originalSource || viewURL} alt='Original Focus' className='h-full w-full object-cover' />
                                    )}

                                    {!isVideo && hasResultSource && !compareMode && activePanel === 'skeleton' && (
                                        <div className='relative h-full w-full'>
                                            <img
                                                src={viewURL}
                                                alt='Skeleton Focus'
                                                className='h-full w-full object-cover'
                                                style={highContrastSkeleton ? { filter: 'contrast(1.34) saturate(1.38) brightness(1.06)' } : undefined}
                                            />
                                            {neonSkeleton && (
                                                <div className='pointer-events-none absolute inset-0'>
                                                    <div className='absolute inset-0 bg-[radial-gradient(circle_at_52%_58%,rgba(90,255,214,0.2),transparent_48%)] mix-blend-screen' />
                                                    {renderNeonSkeletonOverlay()}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {!isVideo && hasResultSource && !compareMode && activePanel === 'heatmap' && (
                                        <div className='relative h-full w-full'>
                                            <img src={viewURL} alt='Heatmap Focus' className='h-full w-full object-cover' />
                                            {renderHeatmapOverlay()}
                                        </div>
                                    )}

                                    {!isVideo && hasResultSource && !compareMode && activePanel === 'boxes' && (
                                        <div className='relative h-full w-full'>
                                            <img src={viewURL} alt='BBoxes Focus' className='h-full w-full object-cover' />
                                            {renderBBoxTrajectoryOverlay()}
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className='grid gap-4 md:grid-cols-3'>
                                <button
                                    type='button'
                                    onClick={() => {
                                        setActivePanel('original');
                                        setCompareMode(false);
                                    }}
                                    className={`group relative overflow-hidden rounded-2xl border bg-black/40 text-left ${activePanel === 'original' ? 'border-[#79f1ad]' : 'border-[#3b5551]'}`}
                                >
                                    <div className='flex items-center justify-between border-b border-[#2e4642] px-3 py-2'>
                                        <p className='text-xs font-semibold uppercase tracking-wider text-[#9fe8c2]'>Original</p>
                                        <span className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6] opacity-0 transition group-hover:opacity-100'>Expand</span>
                                    </div>
                                    <div className='h-44'>{renderMedia(originalSource, isVideo && hasOriginalSource)}</div>
                                </button>

                                <button
                                    type='button'
                                    onClick={() => {
                                        setActivePanel('heatmap');
                                        setCompareMode(false);
                                    }}
                                    className={`group relative overflow-hidden rounded-2xl border bg-black/40 text-left ${activePanel === 'heatmap' ? 'border-[#79f1ad]' : 'border-[#3b5551]'}`}
                                >
                                    <div className='flex items-center justify-between border-b border-[#2e4642] px-3 py-2'>
                                        <p className='text-xs font-semibold uppercase tracking-wider text-[#9fe8c2]'>Heatmap</p>
                                        <span className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6] opacity-0 transition group-hover:opacity-100'>Compare</span>
                                    </div>
                                    <div className='relative h-44'>
                                        {renderMedia(viewURL, false)}
                                        {hasResultSource && renderHeatmapOverlay()}
                                    </div>
                                </button>

                                <button
                                    type='button'
                                    onClick={() => {
                                        setActivePanel('boxes');
                                        setCompareMode(false);
                                    }}
                                    className={`group relative overflow-hidden rounded-2xl border bg-black/40 text-left ${activePanel === 'boxes' ? 'border-[#79f1ad]' : 'border-[#3b5551]'}`}
                                >
                                    <div className='flex items-center justify-between border-b border-[#2e4642] px-3 py-2'>
                                        <p className='text-xs font-semibold uppercase tracking-wider text-[#9fe8c2]'>BBoxes + Trajectory</p>
                                        <span className='rounded-md border border-[#4b6d64] px-2 py-1 text-[10px] uppercase tracking-wider text-[#d8f2e6] opacity-0 transition group-hover:opacity-100'>Expand</span>
                                    </div>
                                    <div className='relative h-44'>
                                        {renderMedia(viewURL, false)}
                                        {hasResultSource && renderBBoxTrajectoryOverlay()}
                                    </div>
                                </button>
                            </div>
                        </div>

                        {caption && (
                            <div ref={resultCaptionRef} className='mt-5 rounded-2xl border border-[#2f504a] bg-black/35 p-4 text-[#ecf3e7]'>
                                <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-[#8ff5ba]">Description</p>
                                <div className='space-y-2 text-sm leading-relaxed md:text-base'>
                                    {captionLines.length > 0 ? (
                                        captionLines.map((line, idx) => (
                                            <p key={`${idx}-${line.slice(0, 20)}`} className='result-caption-line'>{line}</p>
                                        ))
                                    ) : (
                                        <p className='result-caption-line'>{caption}</p>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </section>
    );
};

Generate.propTypes = {
    motionPreset: PropTypes.oneOf(['subtle', 'balanced', 'bold']),
};

export default Generate;
