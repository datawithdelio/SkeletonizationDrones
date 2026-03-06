const NavBar = () => {
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-[#2f3a38] bg-[#03090b]/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1200px] items-center justify-between px-5 py-4">
        <h1 className="font-['Space_Grotesk'] text-2xl font-bold tracking-tight text-[#f3f5de] md:text-4xl">
          Skel
          <span className="bg-[linear-gradient(90deg,#e0ea94_0%,#8cf5d2_55%,#09cb79_100%)] bg-clip-text text-transparent">
            AI
          </span>
        </h1>
        <a
          href="https://github.com/gavw11/Skeleton-Generator-Research-Project/tree/main"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-full border border-[#72f3a9] px-5 py-2 text-xs font-medium uppercase tracking-wider text-[#d6fbe7] transition-all duration-300 hover:bg-[#0de47a] hover:text-black md:text-sm"
        >
          About Project
        </a>
      </div>
    </nav>
  )
}

export default NavBar
