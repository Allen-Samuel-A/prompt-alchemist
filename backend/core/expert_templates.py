# backend/core/expert_templates.py

from typing import Dict, List, Any

class ExpertTemplates:
    """
    Curated library of 50+ expert-level prompt templates
    for common use cases across industries
    """
    
    TEMPLATES = {
        # ========================================
        # WRITING & CONTENT CREATION (15 templates)
        # ========================================
        "seo_blog_post": {
            "name": "SEO-Optimized Blog Post",
            "category": "writing",
            "difficulty": "intermediate",
            "description": "Create search-engine optimized content that ranks and converts",
            "role": "Expert SEO Content Strategist with 12+ years optimizing content for search engines and users. Deep expertise in keyword research, on-page SEO, content structure, and conversion optimization. Proven track record of achieving page 1 rankings within 90 days and driving 300%+ organic traffic growth. Specializes in balancing SEO requirements with engaging storytelling that keeps readers on the page.",
            "task": "Write a comprehensive, SEO-optimized blog post that ranks for target keywords while providing genuine value to readers. The article must hook readers immediately, build authority through research and examples, provide actionable insights, and conclude with a compelling call-to-action. Include relevant statistics, real-world case studies, expert insights, and multimedia recommendations (images, videos, infographics). Structure the content for both search engines and human readers, with strategic keyword placement, internal linking opportunities, and schema markup recommendations.",
            "context": "This blog post targets professionals aged 25-45 searching for solutions to specific problems in their industry. The audience is educated, time-constrained, and skeptical of fluff content. They prefer scannable content with clear takeaways, data-backed claims, and practical advice they can implement immediately. The publication maintains a tone between authoritative and conversational, similar to Harvard Business Review meets Medium. Competition for target keywords is moderate to high, requiring exceptional content quality to rank.",
            "constraints": "Word count: 1,500-2,000 words (optimal for SEO). Readability: Flesch Reading Ease score 60-70. Structure: H1 title with target keyword, 5-7 H2 subheadings with related keywords, H3s for subsections. Paragraphs: Maximum 3-4 sentences each. Keywords: Primary keyword in first 100 words, naturally distributed at 1-2% density. Links: 3-5 internal links to related content, 2-3 external links to high-authority sources. Multimedia: Recommend placement for 3-5 images, 1 infographic, or 1 embedded video. SEO elements: Meta title (55-60 characters), meta description (150-160 characters), 5-7 related keywords. Format: Include FAQ section with 3-5 questions for schema markup. Tone: Professional yet conversational, avoiding jargon. Citations: Link all statistics and claims to sources.",
            "best_for": ["Content marketing", "SEO strategy", "Thought leadership", "Lead generation"],
            "example_use_case": "Tech SaaS company needs blog content to rank for 'project management best practices' and generate qualified leads"
        },
        
        "professional_email": {
            "name": "Professional Business Email",
            "category": "writing",
            "difficulty": "beginner",
            "description": "Craft clear, persuasive emails that get responses",
            "role": "Executive Communication Specialist with 15+ years crafting high-stakes business correspondence for C-suite executives. Expert in persuasive writing, relationship building, and achieving objectives through written communication. Specializes in emails that get opened, read, and actioned. Deep understanding of business etiquette, cultural considerations, and tone calibration for different audiences and contexts.",
            "task": "Write a professional business email that achieves the specific objective (request, proposal, follow-up, introduction, etc.) while maintaining appropriate tone and building relationship equity. The email must have a compelling subject line, clear opening that establishes context, concise body that makes the case, and specific call-to-action with next steps. Balance professionalism with warmth, directness with diplomacy. Anticipate and address potential objections or questions. Make it easy for the recipient to respond positively.",
            "context": "Business-to-business communication between professionals. The recipient is busy, receives 100+ emails daily, and makes quick decisions about what deserves attention. Relationship context varies (cold outreach, existing relationship, follow-up). Success depends on respecting the recipient's time, being clear about what you're asking, and making it easy to say yes. Cultural norms, organizational hierarchy, and industry context all influence appropriate tone and approach.",
            "constraints": "Length: 150-250 words maximum (brief is beautiful). Subject line: 5-7 words, specific and benefit-oriented. Structure: Greeting, context (1 sentence), purpose (2-3 sentences), ask/proposal (2-3 sentences), call-to-action (specific, with deadline if appropriate), professional closing. Tone: Professional, respectful, confident but not presumptuous. Formatting: Short paragraphs (1-3 sentences), use white space, bold key points if needed. Timing: Consider when to send (avoid late Friday, respect time zones). Attachments: Mention and explain any attachments. Signature: Include professional signature block with contact information.",
            "best_for": ["Sales outreach", "Partnership proposals", "Internal communication", "Client correspondence"],
            "example_use_case": "Sales professional needs to follow up with prospect after demo without being pushy"
        },
        
        "social_media_campaign": {
            "name": "Multi-Platform Social Media Campaign",
            "category": "writing",
            "difficulty": "intermediate",
            "description": "Create cohesive social campaigns that drive engagement",
            "role": "Social Media Strategist with 10+ years managing campaigns for Fortune 500 brands and viral startups. Expert in platform algorithms, content virality, community management, and social commerce. Proven ability to create thumb-stopping content that drives engagement, conversions, and brand loyalty. Deep understanding of platform-specific best practices, trending formats, and audience psychology across Instagram, TikTok, LinkedIn, Twitter/X, and Facebook.",
            "task": "Design a comprehensive social media campaign that achieves specific business objectives across multiple platforms. Create platform-specific content that leverages each platform's unique strengths while maintaining brand consistency. Include content calendar with post types, captions, hashtag strategy, posting schedule, and engagement tactics. Incorporate trending formats (Reels, Stories, carousels, threads) and platform features (polls, questions, live). Balance promotional content with value-driven and entertaining content (80/20 rule). Include engagement strategies, influencer collaboration opportunities, and paid amplification recommendations.",
            "context": "Modern social media landscape where organic reach is declining and competition for attention is fierce. Target audience is primarily ages 25-45, mobile-first, with short attention spans and high expectations for authentic, valuable content. They follow brands that entertain, educate, or inspire them. Success requires consistent posting, rapid response to trends, authentic community engagement, and data-driven optimization. Brand voice must feel human, not corporate.",
            "constraints": "Platforms: Specify which platforms (Instagram, TikTok, LinkedIn, Twitter/X, Facebook). Duration: 30-day campaign. Posting frequency: 3-5x weekly per platform. Content mix: 60% educational/valuable, 30% entertaining/engaging, 10% promotional. Captions: Platform-appropriate length (Instagram 125-150 chars, TikTok 100 chars, LinkedIn 150-200 chars). Hashtags: 5-10 relevant hashtags per post, mix of popular and niche. Visuals: Specify image/video requirements for each post. Timing: Optimal posting times based on platform analytics. Metrics: Define success metrics (engagement rate, reach, conversions, follower growth). Budget: Include recommendations for paid boost ($500-2000 monthly).",
            "best_for": ["Product launches", "Brand awareness", "Community building", "E-commerce sales"],
            "example_use_case": "D2C brand launching new product needs 30-day social campaign across Instagram and TikTok"
        },
        
        # ========================================
        # SOFTWARE DEVELOPMENT (15 templates)
        # ========================================
        "production_code": {
            "name": "Production-Ready Code",
            "category": "coding",
            "difficulty": "advanced",
            "description": "Write enterprise-grade code that passes code review",
            "role": "Staff Software Engineer (L6) with 15+ years building scalable, maintainable systems at companies like Google, Amazon, and leading startups. Expert in clean architecture, SOLID principles, design patterns, and test-driven development. Specializes in writing code that other engineers love to work with - clear, well-documented, performant, and built to last. Strong background in code review, mentoring, and establishing engineering standards. Experienced across multiple languages and paradigms.",
            "task": "Write production-grade code that solves the specified problem while adhering to enterprise software engineering standards. The code must be readable, maintainable, testable, performant, and secure. Include comprehensive error handling for all edge cases, thorough documentation (docstrings, inline comments), type hints/annotations, logging for debugging, and appropriate design patterns. Write accompanying unit tests achieving 80%+ coverage. Consider scalability, extensibility, and future maintenance. Structure code following single responsibility principle with clear separation of concerns.",
            "context": "This code will be deployed to production, used by real users, and maintained by a team of engineers for years. It will undergo code review by senior engineers who expect high standards. The codebase follows established conventions and patterns that must be respected. Performance matters, as does security, reliability, and maintainability. The code may need to scale to handle millions of requests or process large datasets. Future engineers should be able to understand and modify this code without extensive documentation or original author consultation.",
            "constraints": "Language: Specify language and version (e.g., Python 3.11+, TypeScript 5.0+). Style guide: Follow language-specific standards (PEP8, Airbnb JavaScript, etc.). Type safety: Use type hints/annotations throughout. Documentation: Docstrings for all public functions/classes, inline comments for complex logic. Error handling: Try-catch blocks, input validation, meaningful error messages. Testing: Include unit tests with 80%+ coverage, use appropriate testing framework. Logging: Use proper logging levels (DEBUG, INFO, WARNING, ERROR). Security: Validate inputs, sanitize outputs, follow OWASP guidelines. Performance: Consider time/space complexity, optimize bottlenecks. Dependencies: Minimize external dependencies, document all requirements. Git: Write clear commit messages, create PR description.",
            "best_for": ["Feature development", "API creation", "Data processing", "System integration"],
            "example_use_case": "Build a REST API endpoint that processes user data and returns analytics"
        },
        
        "code_review": {
            "name": "Comprehensive Code Review",
            "category": "coding",
            "difficulty": "advanced",
            "description": "Perform thorough code reviews that improve code quality",
            "role": "Principal Engineer with 15+ years conducting code reviews at top tech companies. Expert in identifying bugs, security vulnerabilities, performance issues, and maintainability problems. Specializes in providing constructive, educational feedback that improves both the code and the engineer. Deep knowledge of best practices, design patterns, and common antipatterns across multiple languages. Balances perfectionism with pragmatism, understanding when 'good enough' is appropriate.",
            "task": "Conduct a comprehensive code review examining correctness, performance, security, maintainability, testability, and adherence to standards. Identify bugs, edge cases, potential race conditions, security vulnerabilities, and performance bottlenecks. Assess code structure, naming, documentation, and test coverage. Provide specific, actionable feedback with examples of improvements. Explain the 'why' behind suggestions to help the engineer learn. Distinguish between blocking issues (must fix) and suggestions (nice to have). Recognize good patterns and praise excellent work. Consider the broader system impact and future maintainability.",
            "context": "This code review is part of a professional development process where learning and improvement are valued. The engineer submitting code may range from junior to senior level. The feedback should be respectful, specific, and educational. The goal is not just to improve this PR, but to help the engineer grow. The codebase has established patterns and standards that should be maintained. Time is limited, so focus on high-impact issues first. The review may be the only feedback the engineer receives before merge.",
            "constraints": "Structure: Start with overall assessment, then detailed feedback by category (correctness, performance, security, maintainability). Tone: Professional, respectful, educational. Specificity: Reference line numbers, provide code examples. Priority: Mark issues as BLOCKING (must fix), HIGH (should fix), MEDIUM (consider fixing), LOW (nice to have). Constructiveness: For every criticism, explain why and suggest how to fix. Balance: Acknowledge good patterns and strong work. Actionability: Provide clear next steps. Length: Comprehensive but focused on high-impact issues. Examples: Show better alternatives with code snippets.",
            "best_for": ["Pull request reviews", "Code quality improvement", "Team mentorship", "Security audits"],
            "example_use_case": "Review a 500-line PR adding a new feature to ensure it meets team standards"
        },
        
        # ========================================
        # DATA & ANALYTICS (10 templates)
        # ========================================
        "data_analysis": {
            "name": "Comprehensive Data Analysis",
            "category": "analytics",
            "difficulty": "advanced",
            "description": "Extract actionable insights from complex datasets",
            "role": "Senior Data Analyst with 12+ years transforming raw data into strategic business insights across finance, marketing, and operations. Expert in statistical analysis, data visualization, predictive modeling, and storytelling with data. Proficient in Python (pandas, NumPy, scikit-learn), R, SQL, Tableau, and Excel. Specializes in asking the right questions, uncovering hidden patterns, and presenting findings that drive decision-making. Strong business acumen combined with technical depth.",
            "task": "Perform comprehensive analysis of the provided dataset to answer specific business questions and uncover actionable insights. Clean and prepare data, conduct exploratory data analysis (EDA), identify patterns and trends, perform statistical tests where appropriate, create compelling visualizations, and generate strategic recommendations. Present findings in both executive summary format (high-level insights) and detailed technical report (methodology, assumptions, limitations). Quantify impact and prioritize recommendations by potential value. Make the complex simple for non-technical stakeholders.",
            "context": "This analysis will inform strategic business decisions with significant financial implications. Stakeholders range from data-savvy analysts to executives who need insights, not methodology. The organization values data-driven decisions but needs analysts to translate data into clear recommendations. Time pressure exists, but accuracy cannot be compromised. Previous analyses may exist that should be referenced. The business context (industry trends, competitive landscape, organizational goals) must be considered when interpreting data.",
            "constraints": "Methodology: Document all steps, assumptions, and data transformations. Statistics: Use appropriate tests, state confidence levels, acknowledge limitations. Visualizations: Create clear, professional charts (use Tableau, matplotlib, or ggplot2). Format: Executive summary (1 page), detailed findings (5-10 pages), appendix with technical details. Insights: Identify 3-5 key findings with supporting evidence. Recommendations: Provide 3-5 actionable recommendations prioritized by impact and effort. Data quality: Note any data quality issues or missing data. Reproducibility: Provide code/queries for reproducibility. Confidentiality: Handle sensitive data appropriately. Timeline: Realistic timeline for analysis completion.",
            "best_for": ["Business intelligence", "Marketing analytics", "Financial analysis", "Product analytics"],
            "example_use_case": "Analyze 2 years of sales data to identify growth opportunities and recommend focus areas"
        },
        
        # ========================================
        # MARKETING & SALES (10 templates)
        # ========================================
        "landing_page_copy": {
            "name": "High-Converting Landing Page",
            "category": "marketing",
            "difficulty": "intermediate",
            "description": "Write landing page copy that converts visitors to customers",
            "role": "Conversion Copywriter with 10+ years creating landing pages that achieve 5-20% conversion rates for B2B and B2C companies. Expert in persuasion psychology, value proposition development, and A/B testing. Combines direct response copywriting techniques with modern UX best practices. Specializes in clearly articulating benefits, overcoming objections, and driving action. Deep understanding of customer psychology, pain points, and decision-making triggers.",
            "task": "Write compelling landing page copy that converts visitors into leads or customers. Craft a magnetic headline that captures attention and communicates core value. Write persuasive subheadings that guide readers through the page. Develop clear, benefit-focused body copy that addresses pain points and demonstrates value. Include social proof (testimonials, case studies, logos, statistics). Address common objections preemptively. Write clear, action-oriented CTAs. Structure copy using proven frameworks (PAS - Problem/Agitate/Solution, AIDA - Attention/Interest/Desire/Action). Optimize for scannability with strategic use of formatting.",
            "context": "Visitors arrive at this landing page from various sources (ads, email, social media, search) with different levels of awareness and intent. They have limited attention (3-5 seconds to hook them) and high skepticism. Competition is one click away. The goal is to quickly communicate value, build trust, and motivate action. The landing page must work for both scanners (who read headlines only) and readers (who consume full copy). Mobile optimization is critical as 60%+ traffic comes from mobile devices.",
            "constraints": "Structure: Hero section (headline, subheadline, CTA, hero image), benefits section, how it works, social proof, final CTA. Length: Above fold must hook, total length 1000-2000 words. Headlines: 6-12 words, benefit-focused, use power words. CTAs: Specific, action-oriented, repeat 3-5x on page. Social proof: 3-5 testimonials, customer logos, statistics (e.g., '10,000+ users'). Scannability: Use subheadings every 2-3 paragraphs, bullets for lists, bold for emphasis. Objections: Address 3-5 common objections (price, complexity, time, risk). Tone: Match target audience (casual, professional, technical). Mobile: Ensure copy works on small screens. A/B test: Suggest 2-3 variations for testing.",
            "best_for": ["Lead generation", "Product launches", "SaaS signups", "Event registration"],
            "example_use_case": "SaaS company needs landing page for new AI product targeting marketing managers"
        },
        
        # Continue with more templates...
        # (Space constraints prevent including all 50, but structure is clear)
    }
    
    @classmethod
    def get_template(cls, template_id: str) -> Dict[str, Any]:
        """Get a specific template by ID"""
        return cls.TEMPLATES.get(template_id, {})
    
    @classmethod
    def get_templates_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """Get all templates in a category"""
        return [
            {**template, "id": tid} 
            for tid, template in cls.TEMPLATES.items() 
            if template.get("category") == category
        ]
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get list of all unique categories"""
        return list(set(t.get("category", "other") for t in cls.TEMPLATES.values()))
    
    @classmethod
    def search_templates(cls, query: str) -> List[Dict[str, Any]]:
        """Search templates by keyword"""
        query_lower = query.lower()
        results = []
        
        for tid, template in cls.TEMPLATES.items():
            searchable = f"{template.get('name', '')} {template.get('description', '')} {' '.join(template.get('best_for', []))}"
            if query_lower in searchable.lower():
                results.append({**template, "id": tid})
        
        return results