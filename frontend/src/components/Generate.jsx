import { useState, useRef, useEffect } from 'react';
import loadingSpinner from '../assets/loading_spinner.svg';
import sendButton from '../assets/send.svg';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const Generate = () => {
    const [downloadReady, setDownloadReady] = useState(false);
    const [viewURL, setViewURL] = useState(null);
    const [file, setFile] = useState(null);
    const [fileURL, setFileURL] = useState('#');
    const [generating, setGenerating] = useState(false);
    const [generateDisabled, setGenerateDisabled] = useState(false);
    const [isImage, setIsImage] = useState(null);
    const [isVideo, setIsVideo] = useState(null);
    const [prompt, setPrompt] = useState(null);
    const [caption, setCaption] = useState(null);
    const [dataURL, setDataURL] = useState(null);
    const [errorMessage, setErrorMessage] = useState(null);
    const [generationSettings, setGenerationSettings] = useState({
        confidence_level: 0.5,
        smoothing_factor: 7,
        downsample: 1
    });

    const inputFileRef = useRef(null);
    const videoRef = useRef(null);

    const handleSettingsChange = (event) => {
        const { name, value } = event.target;
        setGenerateDisabled(false);
        setGenerationSettings(prev => ({
            ...prev,
            [name]: parseFloat(value)
        }));
    };

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.load();
        }
    }, [viewURL]);

    const handlePromptChange = (e) => {
        setPrompt(e.target.value);
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const type = file.type;
            setIsImage(type.startsWith('image/'));
            setIsVideo(type.startsWith('video/'));
            setGenerateDisabled(false);
            setFile(file);
        }
    };

    const handleInput = () => inputFileRef.current.click();

    const handleAiGenerate = async () => {
        if (!prompt) return alert("No Prompt!");
        setIsImage(true);
        setIsVideo(false);
        setGenerateDisabled(true);
        setGenerating(true);
        setErrorMessage(null);

        try {
            const res = await axios.post(`${API_BASE_URL}/api/openai/upload`, {
                prompt,
                generationSettings
            });

            const id = res.data.id;
            setFileURL(`${API_BASE_URL}/api/download/${id}`);
            setDataURL(`${API_BASE_URL}/api/download_data/${id}`);
            setViewURL(`${API_BASE_URL}/api/view/${id}`);
            setCaption(res.data.caption);
            setDownloadReady(true);
        } catch (err) {
            console.error('Error uploading prompt:', err);
            setDownloadReady(false);
            setErrorMessage(err.response?.data?.msg || 'AI generation failed.');
        } finally {
            setGenerating(false);
        }
    };

    const handleGenerate = async () => {
        if (!file) return;

        setGenerateDisabled(true);
        setGenerating(true);
        setErrorMessage(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileName', file.name);
        formData.append('generationSettings', JSON.stringify(generationSettings));

        try {
            const res = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            const id = res.data.id;
            setFileURL(`${API_BASE_URL}/api/download/${id}`);
            setDataURL(`${API_BASE_URL}/api/download_data/${id}`);
            setViewURL(`${API_BASE_URL}/api/view/${id}`);
            setCaption(res.data.caption);
            setDownloadReady(true);
        } catch (err) {
            console.error('Error uploading file:', err);
            setDownloadReady(false);
            setErrorMessage(err.response?.data?.msg || 'File upload failed.');
        } finally {
            setGenerating(false);
        }
    };

    return (
        <>
            <div className='w-full h-14 pt-28'>
                {generating && (
                    <div className='flex flex-col items-center justify-center'>
                        <img className='w-10 h-10' src={loadingSpinner} alt="Loading..." />
                        <p>Generating Skeleton...</p>
                    </div>
                )}
                {errorMessage && (
                    <div className='flex justify-center px-4'>
                        <p className='rounded-xl border border-red-500 bg-red-950/40 px-4 py-2 text-center text-sm text-red-200 md:text-base'>
                            {errorMessage}
                        </p>
                    </div>
                )}
            </div>

            <div className='flex flex-row w-full h-full justify-center pt-24'>
                <input type='file' onChange={handleFileChange} className='hidden' ref={inputFileRef} />
                <div className='flex justify-center items-center px-10'>
                    <button onClick={handleInput} className='px-1 py-2 h-12 mx-5 bg-black border border-green-600 hover:bg-green-600 rounded-2xl text-xs shadow-md transition-all duration-500 font-light md:text-2xl md:mx-12 md:px-6'>
                        Upload File
                    </button>
                    <button onClick={handleGenerate} disabled={generateDisabled} className='px-1 h-12 mx-5 bg-green-500 hover:bg-green-700 rounded-2xl text-xs font-light transition-all duration-500 md:text-2xl md:mx-12 md:px-6'>
                        Generate Skeleton
                    </button>
                </div>
            </div>

            <div className='flex flex-col w-full h-full justify-center items-center pt-14'>
                <h2 className='font-normal text-xl'>Generate Image + Skeleton With AI:</h2>
                <div className='flex flex-row items-center justify-center mt-3'>
                    <input type='text' onChange={handlePromptChange} placeholder='Enter Image Prompt...' className='border rounded-xl border-white w-96 h-10 px-5 font-light mr-2' />
                    <button className='bg-green-400 rounded-xl hover:bg-green-700 transition-all duration-500 w-10 h-11 inline-flex items-center justify-center' onClick={handleAiGenerate}>
                        <img src={sendButton} className='w-6 h-6' alt="Send" />
                    </button>
                </div>
            </div>

            <div className='flex flex-col justify-center items-center w-full pt-10'>
                <label>Confidence Level:
                    <input className='mx-4' type='range' name='confidence_level' min='0.1' max='1' step='0.05' value={generationSettings.confidence_level} onChange={handleSettingsChange} />
                    {generationSettings.confidence_level}
                </label>
                <label>Smoothing Factor:
                    <input className='mx-4' type='range' name='smoothing_factor' min='1' max='30' step='1' value={generationSettings.smoothing_factor} onChange={handleSettingsChange} />
                    {generationSettings.smoothing_factor}
                </label>
                <label>Downsample Factor:
                    <input className='mx-4' type='range' name='downsample' min='1' max='8' step='1' value={generationSettings.downsample} onChange={handleSettingsChange} />
                    {generationSettings.downsample}
                </label>
            </div>

            {downloadReady && (
                <div className='flex flex-col justify-center items-center w-full h-full pt-10'>
                    <div className='flex flex-row justify-center items-center w-full h-full'>
                        <a href={fileURL} className='inline-flex items-center justify-center px-3 h-12 mx-12 bg-black border border-white hover:bg-white hover:text-black rounded-2xl text-2xl shadow-md transition-all duration-700 font-light'>
                            Download Skeleton
                        </a>
                        <a href={dataURL} className='inline-flex items-center justify-center px-3 h-12 mx-12 bg-black border border-white hover:bg-white hover:text-black rounded-2xl text-2xl shadow-md transition-all duration-700 font-light'>
                            Download Point Data
                        </a>
                    </div>
                    {isVideo ? (
                        <video autoPlay loop ref={videoRef} className='mt-10 w-1/2 h-auto rounded-md border-4 border-green-400 object-cover'>
                            <source src={viewURL} type='video/mp4' />
                        </video>
                    ) : (
                        <img src={viewURL} className='mt-10 w-1/2 h-auto rounded-md border-4 border-green-400 object-cover' alt="Skeleton" />
                    )}
                    {caption && (
                        <p className='mt-4 px-6 text-lg text-center text-white font-light'>
                            <span className="font-semibold text-green-300">Description:</span> {caption}
                        </p>
                    )}
                </div>
            )}
        </>
    );
};

export default Generate;
