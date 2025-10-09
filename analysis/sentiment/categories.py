CATEGORY_DESCRIPTIONS = {
    # Core Technical Categories
    "systems_programming": "Low-level code, memory management, compilers, OS internals, C, C++, Rust, assembly"
    , "distributed_systems": "Distributed architecture, consensus, replication, microservices, distributed databases"
    , "networking": "Network protocols, TCP/IP, DNS, routing, CDN, packet analysis"
    , "programming_languages": "Language design, type systems, parsers, interpreters, new languages"
    , "algorithms": "Algorithm analysis, data structures, complexity, sorting, graph algorithms"
    , "cli_tools": "Command-line programs, terminal utilities, shell scripts, bash, zsh"

    # Domain-Specific
    , "cryptography": "Encryption, hashing, digital signatures, TLS, zero-knowledge proofs, cryptographic protocols"
    , "data_engineering": "ETL pipelines, data warehouses, Spark, Kafka, stream processing, data infrastructure"
    , "hardware": "Processors, chips, circuits, semiconductors, electronics, embedded devices"
    , "mobile_development": "iOS apps, Android apps, mobile frameworks, Swift, Kotlin, React Native"
    , "game_development": "Game engines, graphics, shaders, Unity, Unreal, game physics"
    , "scientific_computing": "Numerical methods, simulations, high-performance computing, scientific software"

    # Industry & Business
    , "robotics": "Physical robots, autonomous vehicles, robot control, actuators, ROS, manipulation"
    , "tech_career": "Job hunting, interviews, salaries, workplace advice, hiring practices"
    , "saas": "SaaS products, B2B software, subscription models, SaaS business metrics"
    , "legal_tech": "Tech law, software patents, open source licenses, GDPR, privacy law"
    , "tech_culture": "Tech industry culture, workplace dynamics, diversity, tech ethics"
    , "productivity": "Task management, note apps, workflows, time management, PKM"

    # Emerging Tech
    , "quantum_computing": "Quantum bits, quantum gates, quantum algorithms, quantum processors"
    , "ar_vr": "Virtual reality, augmented reality, VR headsets, spatial computing, 3D interfaces"
    , "iot": "Connected devices, IoT sensors, smart home, edge devices, embedded networking"
    , "sustainability": "Clean tech, renewable energy, carbon reduction, e-waste, green computing"

    # Content Types
    , "research_paper": "Academic research, peer-reviewed papers, arxiv preprints, scientific studies"
    , "historical": "Computing history, vintage computers, retro tech, legacy systems, tech archaeology"
    , "tutorial": "How-to guides, step-by-step instructions, learning materials, educational walkthroughs"
    , "opinion": "Opinion essays, commentary, thought pieces, editorials, tech punditry"
    , "pop_culture": "Entertainment, movies, TV shows, gaming culture, internet memes, viral content"
    , "general_news": "Current events, breaking news, acquisitions, major announcements, politics"
    , "research_news": "Scientific discoveries, research breakthroughs, Nobel prizes, academic achievements"

    # Platform-specific
    , "linux": "Linux distros, kernel, Unix systems, sysadmin, GNU tools"
    , "macos_ios": "macOS, iOS, Apple platforms, Swift, Xcode, Apple hardware"
    , "windows": "Windows OS, .NET, Visual Studio, PowerShell, Microsoft technologies"
    , "cloud_platforms": "AWS, Azure, GCP, cloud services, serverless, cloud architecture"

    # Specialized Areas
    , "monitoring_observability": "Logs, metrics, traces, monitoring tools, debugging, APM"
    , "testing_qa": "Unit tests, integration tests, test automation, TDD, code coverage"
    , "documentation": "Technical docs, API documentation, readme files, documentation generators"
    , "accessibility": "A11y, WCAG, screen readers, accessible design, inclusive interfaces"
    , "performance": "Speed optimization, profiling, benchmarks, performance tuning, latency"
    , "ui_ux": "Interface design, user experience, design systems, prototypes, wireframes"
    , "api_design": "REST, GraphQL, API patterns, endpoint design, API documentation"
    , "developer_tools": "Code editors, IDE, vim, git, build tools, developer utilities"

    # Language-specific
    , "python_related": "Python packages, pip, Django, Flask, pandas, numpy"
    , "react_related": "React components, hooks, Next.js, React ecosystem, JSX"
    , "rust_related": "Rust crates, cargo, lifetimes, borrow checker, Rust projects"
    , "go_related": "Go packages, goroutines, Go standard library, Go projects"
    , "c_related": "C code, C++, C#, pointers, manual memory, C libraries"
    , "ruby_rails_related": "Ruby gems, Rails apps, Rack, Sinatra, RSpec, ActiveRecord, Ruby projects"

    # Specialized Topics
    , "economic_analysis": "Economics, markets, inflation, monetary policy, financial analysis"
    , "security_vulnerability": "CVE, exploits, security bugs, vulnerabilities, security advisories"
    , "technical_blog": "Engineering blogs, tech write-ups, developer posts, technical explanations"
    , "startup_news": "Startup funding, YC companies, venture capital, startup launches"
    , "ai_ml": "AI, Neural networks, transformers, LLMs, deep learning, machine learning models, OpenAI, Gemini, Claude, Grok"
    , "web_development": "HTML, CSS, JavaScript, frontend frameworks, web servers, HTTP"
    , "devops": "CI/CD, Docker, Kubernetes, infrastructure as code, deployment automation"
    , "database": "SQL, NoSQL, PostgreSQL, MySQL, MongoDB, query performance, database design"
    , "open_source": "OSS projects, GitHub, open source licenses, FOSS, contributor communities"

    # Catch-all
    , "other_technical": "Software engineering, coding, programming, technical topics"
    , "general_tech_news": "Tech industry news, company updates, tech sector announcements"
}

CATEGORIES = list(CATEGORY_DESCRIPTIONS.keys())

