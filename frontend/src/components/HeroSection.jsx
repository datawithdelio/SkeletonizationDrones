import { useLayoutEffect, useMemo, useRef, useState } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import VideoBackground from "./VideoBackground"
import Generate from "./Generate"
import NavBar from "./NavBar"

const HeroSection = () => {
  const sectionRef = useRef(null)
  const tickerRef = useRef(null)
  const heroPanelRef = useRef(null)
  const heroBadgeRef = useRef(null)
  const heroTitleRef = useRef(null)
  const heroDescriptionRef = useRef(null)
  const generateRef = useRef(null)
  const glowLeftRef = useRef(null)
  const glowRightRef = useRef(null)
  const [motionPreset, setMotionPreset] = useState("balanced")
  const motionConfig = useMemo(
    () =>
      ({
        subtle: { introYOffset: 18, heroScrubShift: -4, glowScrubShift: -10, blurScale: 0.9, tickerDuration: 22 },
        balanced: { introYOffset: 30, heroScrubShift: -8, glowScrubShift: -20, blurScale: 1, tickerDuration: 18 },
        bold: { introYOffset: 44, heroScrubShift: -14, glowScrubShift: -30, blurScale: 1.14, tickerDuration: 14 },
      }[motionPreset]),
    [motionPreset]
  )

  useLayoutEffect(() => {
    if (!sectionRef.current) {
      return undefined
    }

    const ctx = gsap.context(() => {
      gsap.registerPlugin(ScrollTrigger)
      const headlineWords = gsap.utils.toArray(".headline-word")

      gsap.set(
        [
          heroBadgeRef.current,
          heroDescriptionRef.current,
          generateRef.current,
        ],
        { autoAlpha: 0, y: motionConfig.introYOffset }
      )
      gsap.set(headlineWords, { autoAlpha: 0, yPercent: 112, rotate: 2.5, transformOrigin: "50% 100%" })
      gsap.set(heroPanelRef.current, { autoAlpha: 0, y: 24, scale: 0.985 })
      gsap.set([glowLeftRef.current, glowRightRef.current], { autoAlpha: 0, scale: 0.85 * motionConfig.blurScale })

      const introTimeline = gsap.timeline({
        defaults: { ease: "power3.out" },
        delay: 0.18,
      })

      introTimeline
        .to(heroPanelRef.current, { autoAlpha: 1, y: 0, scale: 1, duration: 0.85 })
        .to([glowLeftRef.current, glowRightRef.current], { autoAlpha: 1, scale: 1, duration: 1.2 }, "<0.05")
        .to(heroBadgeRef.current, { autoAlpha: 1, y: 0, duration: 0.42 }, "-=0.5")
        .to(headlineWords, { autoAlpha: 1, yPercent: 0, rotate: 0, duration: 0.88, stagger: 0.14 }, "-=0.18")
        .to(heroDescriptionRef.current, { autoAlpha: 1, y: 0, duration: 0.5 }, "-=0.32")
        .to(generateRef.current, { autoAlpha: 1, y: 0, duration: 0.75 }, "-=0.24")

      gsap.to(tickerRef.current, {
        xPercent: -50,
        ease: "none",
        duration: motionConfig.tickerDuration,
        repeat: -1,
      })

      gsap.to(glowLeftRef.current, {
        y: -10,
        x: 5,
        duration: 4.4,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      })

      gsap.to(glowRightRef.current, {
        y: 12,
        x: -8,
        duration: 5.2,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      })

      gsap.to(heroPanelRef.current, {
        yPercent: motionConfig.heroScrubShift,
        scale: 0.985,
        ease: "none",
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top top",
          end: "bottom top",
          scrub: 0.8,
        },
      })

      gsap.to([glowLeftRef.current, glowRightRef.current], {
        yPercent: motionConfig.glowScrubShift,
        ease: "none",
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top top",
          end: "bottom top",
          scrub: 1.1,
        },
      })

      gsap.to(headlineWords, {
        yPercent: motionConfig.heroScrubShift * 1.8,
        ease: "none",
        stagger: 0.08,
        scrollTrigger: {
          trigger: sectionRef.current,
          start: "top top",
          end: "bottom top",
          scrub: 0.7,
        },
      })
    }, sectionRef)

    return () => ctx.revert()
  }, [motionConfig])

  return (
    <section ref={sectionRef} className="zen-hero z-0 w-full">
      <div className="relative z-50 w-full overflow-hidden bg-[linear-gradient(90deg,#9de7ca_0%,#b7e8db_100%)] py-1.5 text-center text-[11px] font-medium tracking-wide text-[#0b2c24] md:text-xs">
        <div ref={tickerRef} className="ticker-track">
          <span>Skeleton Intelligence Studio: Multi-method detection, skeletonization, and trajectory analysis.</span>
          <span aria-hidden="true">Skeleton Intelligence Studio: Multi-method detection, skeletonization, and trajectory analysis.</span>
        </div>
      </div>
      <NavBar />
      <div className="fixed top-0 z-0 w-full">
        <VideoBackground />
      </div>
      <div className="relative z-40 mx-auto flex w-full max-w-[1320px] flex-col px-4 pb-10 pt-5 md:px-8">
        <div ref={glowLeftRef} className="pointer-events-none absolute -top-12 left-[8%] h-52 w-52 rounded-full bg-[#2fffb45e] blur-3xl" />
        <div ref={glowRightRef} className="pointer-events-none absolute right-[4%] top-[28%] h-44 w-44 rounded-full bg-[#9d8bff52] blur-3xl" />
        <div ref={heroPanelRef} className="zen-panel mb-5 rounded-3xl border border-[#254540] bg-[#02080a]/75 p-5 backdrop-blur-sm md:p-7">
          <p ref={heroBadgeRef} className="mb-3 text-[10px] font-semibold uppercase tracking-[0.32em] text-[#91f6bc] animate-pulse">
            Drone Skeletonization Platform
          </p>
          <div className="mb-5 inline-flex rounded-full border border-[#3a5b50] bg-black/25 p-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#d8f4e6]">
            {["subtle", "balanced", "bold"].map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => setMotionPreset(preset)}
                className={`rounded-full px-3 py-1 transition-all ${motionPreset === preset ? "bg-[#1df08e] text-black" : "text-[#c8e8d8] hover:bg-[#143529]"}`}
              >
                {preset}
              </button>
            ))}
          </div>
          <h1 ref={heroTitleRef} className="font-['Space_Grotesk'] text-3xl font-bold leading-[0.95] text-[#f2f4df] md:text-5xl lg:text-6xl text-shadow-hero">
            <span className="headline-mask block">
              <span className="headline-word inline-block">
                Map
                <span className="ml-3 inline-block bg-[linear-gradient(145deg,#ffb33f_0%,#f7a2f9_48%,#8fb8ff_100%)] bg-clip-text text-transparent float-soft">
                  Motion
                </span>
              </span>
            </span>
            <span className="headline-mask block">
              <span className="headline-word inline-block">
                from
                <span className="ml-3 inline-block text-[#afffb6]">Form</span>
              </span>
            </span>
          </h1>
          <p ref={heroDescriptionRef} className="mt-4 max-w-3xl text-xs leading-relaxed text-[#e4eadf] md:text-sm">
            Upload media, run detection + segmentation + skeletonization, and inspect trajectory outputs in a research-grade dashboard.
          </p>
        </div>
        <div ref={generateRef}>
          <Generate motionPreset={motionPreset} />
        </div>
      </div>
    </section>
  )
}

export default HeroSection
