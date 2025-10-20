# backend/core/prompt_enricher.py

from typing import Dict, List, Tuple
import re

class PromptEnricher:
    """
    Automatically enriches basic user input into expert-level prompts
    by adding depth, specificity, and professional context.
    """
    
    # Role enrichment database
    ROLE_ENRICHMENTS = {
        "writer": {
            "keywords": ["writer", "author", "content", "copywriter", "blogger"],
            "expansion": "Expert Content Strategist and Writer with 10+ years of experience crafting high-converting content across multiple formats (blog posts, whitepapers, case studies, landing pages). Specializes in translating complex topics into engaging narratives that drive measurable results, combining SEO expertise with persuasive storytelling techniques. Proven track record of creating content that ranks on page 1 of Google and achieves 3x industry-standard engagement rates."
        },
        "developer": {
            "keywords": ["developer", "programmer", "engineer", "coder", "software"],
            "expansion": "Senior Software Engineer with 12+ years of experience building scalable, production-grade applications. Deep expertise in clean architecture, SOLID principles, design patterns, and test-driven development. Specializes in writing maintainable, well-documented code that passes rigorous code reviews. Experienced with modern frameworks, CI/CD pipelines, and performance optimization. Strong background in system design and mentoring junior developers."
        },
        "marketer": {
            "keywords": ["marketer", "marketing", "campaign", "promotion", "brand"],
            "expansion": "Strategic Digital Marketing Expert with 10+ years driving growth for B2B and B2C companies. Specializes in multi-channel campaigns across SEO, SEM, social media, email, and content marketing. Data-driven approach with expertise in analytics, A/B testing, conversion rate optimization, and marketing attribution. Proven track record of 300%+ ROI improvements and building scalable growth engines. Skilled in both strategy and hands-on execution."
        },
        "analyst": {
            "keywords": ["analyst", "analysis", "data", "research", "insights"],
            "expansion": "Senior Data Analyst with 10+ years transforming complex datasets into actionable business insights. Expert in statistical analysis, predictive modeling, data visualization, and storytelling with data. Proficient in Python, R, SQL, Tableau, and Excel. Specializes in translating technical findings into clear recommendations for non-technical stakeholders. Experience across finance, marketing, operations, and product analytics."
        },
        "designer": {
            "keywords": ["designer", "design", "creative", "visual", "graphic"],
            "expansion": "Senior UX/UI Designer with 10+ years creating user-centered digital experiences that balance aesthetics with functionality. Expert in user research, wireframing, prototyping, and usability testing. Proficient in Figma, Adobe Creative Suite, and design systems. Specializes in converting business goals into intuitive interfaces that drive engagement and conversion. Strong understanding of accessibility, responsive design, and design thinking methodologies."
        },
        "consultant": {
            "keywords": ["consultant", "advisor", "strategist", "expert"],
            "expansion": "Strategic Business Consultant with 15+ years advising Fortune 500 companies and high-growth startups. Expert in business model innovation, operational efficiency, change management, and strategic planning. Combines data-driven analysis with creative problem-solving to deliver transformative results. Proven track record of helping organizations achieve 50-200% revenue growth and streamline operations for maximum efficiency."
        }
    }
    
    # Task enhancement patterns
    TASK_ENHANCEMENTS = {
        "write": {
            "elements": [
                "Define specific deliverable (blog post, article, whitepaper, email, etc.)",
                "Specify target word count or length range",
                "Include key messages or main points to cover",
                "Define tone (professional, casual, authoritative, friendly)",
                "Specify structure (intro, body sections, conclusion, CTA)",
                "Include SEO requirements if applicable",
                "Define success metrics (engagement, conversions, shares)"
            ],
            "template": "Create a {format} that {achieves_goal}, structured with {structure_details}, hitting {key_points}, while maintaining a {tone} tone that resonates with {audience}. Include {specific_elements} and end with a {cta_type}."
        },
        "code": {
            "elements": [
                "Specify programming language and version",
                "Define input parameters and expected outputs",
                "Include error handling requirements",
                "Specify coding standards (PEP8, ESLint, etc.)",
                "Define testing requirements (unit tests, integration tests)",
                "Include documentation standards (docstrings, comments)",
                "Specify design patterns to use or avoid",
                "Define performance requirements"
            ],
            "template": "Write {language} code that {functionality}, accepting {inputs} and returning {outputs}. Include comprehensive error handling for {edge_cases}, follow {coding_standard} guidelines, and provide {test_coverage}% test coverage with {test_types}. Code must be production-ready, well-documented, and optimized for {performance_criteria}."
        },
        "analyze": {
            "elements": [
                "Define data source and type",
                "Specify analysis methodology",
                "Define key metrics to calculate",
                "Specify visualization requirements",
                "Define insight format (executive summary, detailed report)",
                "Include statistical rigor requirements",
                "Define actionable recommendations format"
            ],
            "template": "Analyze {data_type} using {methodology} to uncover {objectives}. Calculate and visualize {key_metrics}, identify {patterns}, and deliver insights in {format}. Include {statistical_elements} and provide {number} actionable recommendations prioritized by {criteria}."
        },
        "create": {
            "elements": [
                "Specify format and medium",
                "Define creative direction and style",
                "Include brand guidelines if applicable",
                "Specify deliverable elements",
                "Define approval process",
                "Include revision expectations"
            ],
            "template": "Create a {deliverable_type} that {purpose}, following {style_guidelines} and incorporating {required_elements}. The final output should be {format_specs} and optimized for {use_case}. Include {number} concepts/variations for review."
        }
    }
    
    # Context enhancement questions
    CONTEXT_QUESTIONS = {
        "audience": [
            "Who is the specific target audience? (age, profession, expertise level)",
            "What are their pain points, goals, and motivations?",
            "What's their current knowledge level on this topic?",
            "Where will they encounter this content? (context of use)",
            "What action do you want them to take?"
        ],
        "business": [
            "What's the business objective or success metric?",
            "What's the competitive landscape?",
            "What resources or constraints exist?",
            "What's the timeline and urgency?",
            "How does this fit into the broader strategy?"
        ],
        "technical": [
            "What systems, tools, or platforms are involved?",
            "What technical constraints exist?",
            "What's the current state vs desired state?",
            "What dependencies or integrations matter?",
            "What performance or scalability requirements exist?"
        ]
    }
    
    # Constraint templates
    CONSTRAINT_TEMPLATES = {
        "quality": [
            "Accuracy: All facts must be verified and cited from credible sources",
            "Originality: Content must be 100% original, no plagiarism",
            "Professionalism: Maintain professional standards throughout"
        ],
        "format": [
            "Length: {min_length} to {max_length} {unit}",
            "Structure: Use clear headings (H1, H2, H3) and short paragraphs (2-3 sentences)",
            "Formatting: Include bullet points for lists, bold for emphasis, links for references"
        ],
        "style": [
            "Tone: {tone_description} (e.g., conversational yet authoritative)",
            "Voice: Use {person} person ({first/second/third})",
            "Language: {language_level} (e.g., simple, technical, academic)",
            "Avoid: {things_to_avoid} (e.g., jargon, passive voice, fluff)"
        ],
        "technical": [
            "Compatibility: Must work with {platforms_versions}",
            "Performance: Must handle {scale_requirements}",
            "Security: Follow {security_standards}",
            "Accessibility: Meet {accessibility_standards} (e.g., WCAG 2.1 AA)"
        ]
    }
    
    @classmethod
    def detect_task_type(cls, text: str) -> str:
        """Detects the primary task type from user input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["write", "blog", "article", "post", "content", "copy"]):
            return "write"
        elif any(word in text_lower for word in ["code", "program", "function", "script", "develop", "build"]):
            return "code"
        elif any(word in text_lower for word in ["analyze", "analysis", "data", "research", "study", "examine"]):
            return "analyze"
        elif any(word in text_lower for word in ["create", "design", "make", "generate", "produce"]):
            return "create"
        else:
            return "general"
    
    @classmethod
    def enrich_role(cls, basic_role: str, task_context: str = "") -> str:
        """
        Transforms a basic role into an expert-level role description
        
        Args:
            basic_role: User's basic role input (e.g., "writer", "developer")
            task_context: Additional context about the task
            
        Returns:
            Enhanced role description with credentials and expertise
        """
        if not basic_role or len(basic_role.strip()) < 2:
            # Detect from task context
            task_type = cls.detect_task_type(task_context)
            if task_type == "write":
                return cls.ROLE_ENRICHMENTS["writer"]["expansion"]
            elif task_type == "code":
                return cls.ROLE_ENRICHMENTS["developer"]["expansion"]
            elif task_type == "analyze":
                return cls.ROLE_ENRICHMENTS["analyst"]["expansion"]
            else:
                return "Expert Professional with extensive experience in delivering high-quality results"
        
        role_lower = basic_role.lower()
        
        # Find matching role enrichment
        for role_key, role_data in cls.ROLE_ENRICHMENTS.items():
            if any(keyword in role_lower for keyword in role_data["keywords"]):
                return role_data["expansion"]
        
        # Fallback: enhance generic role
        return f"Expert {basic_role.title()} with 10+ years of professional experience, deep domain expertise, and a proven track record of delivering exceptional results. Combines technical mastery with strategic thinking and clear communication to solve complex problems effectively."
    
    @classmethod
    def enrich_task(cls, basic_task: str) -> str:
        """
        Transforms a basic task into a detailed, actionable objective
        
        Args:
            basic_task: User's basic task input
            
        Returns:
            Enhanced task with specific deliverables and success criteria
        """
        if not basic_task or len(basic_task.strip()) < 3:
            return "Complete the specified objective with high quality, attention to detail, and alignment with best practices."
        
        task_type = cls.detect_task_type(basic_task)
        
        # Add specificity based on task type
        enhancements = {
            "write": "Create comprehensive, engaging content that achieves the specified goal while maintaining high quality standards. The deliverable should be well-structured, thoroughly researched, and optimized for the target audience. Include clear headings, compelling examples, and actionable takeaways.",
            
            "code": "Develop production-ready, maintainable code that solves the specified problem efficiently. The solution must follow coding best practices, include comprehensive error handling, be well-documented, and come with appropriate tests. Prioritize code clarity, performance, and scalability.",
            
            "analyze": "Conduct thorough analysis using appropriate methodologies to extract meaningful insights from the data. Identify key patterns, trends, and anomalies. Present findings with clear visualizations and provide data-driven recommendations prioritized by potential impact.",
            
            "create": "Design and produce a high-quality deliverable that meets the specified requirements and exceeds stakeholder expectations. Ensure the output is polished, professional, and optimized for its intended use case. Include rationale for key creative decisions."
        }
        
        base_enhancement = enhancements.get(task_type, "Successfully complete the task with attention to quality, accuracy, and user needs.")
        
        return f"{basic_task.strip()} {base_enhancement}"
    
    @classmethod
    def enrich_context(cls, basic_context: str, role: str, task: str) -> str:
        """
        Adds comprehensive background, audience details, and environmental factors
        
        Args:
            basic_context: User's basic context input
            role: The role (for context)
            task: The task (for context)
            
        Returns:
            Enhanced context with audience psychology and use case details
        """
        if not basic_context or len(basic_context.strip()) < 5:
            task_type = cls.detect_task_type(task)
            return f"This work is for a professional audience that values quality, clarity, and practical value. The deliverable will be used in a business/professional context where accuracy and attention to detail are critical. Consider industry best practices and current trends when approaching this task."
        
        # Build enriched context
        enriched = f"{basic_context.strip()}"
        
        # Add audience insights if not present
        if "audience" not in basic_context.lower():
            enriched += " The target audience consists of professionals who value clear, actionable information presented in an accessible format."
        
        # Add success criteria if not present
        if "success" not in basic_context.lower() and "goal" not in basic_context.lower():
            enriched += " Success will be measured by clarity, thoroughness, and practical applicability of the output."
        
        # Add use case context
        enriched += " This deliverable should be immediately usable in a real-world professional setting and aligned with current industry standards and best practices."
        
        return enriched
    
    @classmethod
    def enrich_constraints(cls, basic_constraints: str, task_type: str) -> str:
        """
        Expands basic constraints into comprehensive guidelines
        
        Args:
            basic_constraints: User's basic constraints
            task_type: Type of task being performed
            
        Returns:
            Enhanced constraints with quality standards and detailed requirements
        """
        if not basic_constraints or basic_constraints.lower() in ["none", "no constraints", "n/a"]:
            # Provide sensible defaults based on task type
            defaults = {
                "write": "Length: 800-1200 words. Tone: Professional yet conversational. Structure: Include introduction, clear sections with headers, and conclusion. Format: Use short paragraphs, bullet points for lists, and bold for emphasis. Quality: Proofread for grammar and clarity.",
                
                "code": "Language: Use latest stable version. Standards: Follow language-specific best practices (PEP8, ESLint, etc.). Documentation: Include docstrings and inline comments. Testing: Aim for 80%+ code coverage. Error handling: Handle edge cases gracefully. Performance: Optimize for readability first, then performance.",
                
                "analyze": "Methodology: Use appropriate statistical methods. Visualization: Include clear charts/graphs. Insights: Provide executive summary. Recommendations: Include 3-5 actionable items. Confidence: State confidence levels and limitations.",
                
                "create": "Quality: Professional, polished output. Format: Appropriate for intended use. Brand: Maintain consistency. Revision: Include rationale for key decisions."
            }
            return defaults.get(task_type, "Maintain professional quality standards. Be thorough, accurate, and clear.")
        
        # Enhance existing constraints
        enhanced = f"{basic_constraints.strip()}"
        
        # Add quality standards if not present
        if "quality" not in basic_constraints.lower():
            enhanced += " Quality: Maintain professional standards throughout, with attention to detail and accuracy."
        
        # Add format guidance if not present
        if "format" not in basic_constraints.lower() and task_type == "write":
            enhanced += " Format: Use clear structure with appropriate headings, short paragraphs, and visual hierarchy."
        
        return enhanced
    
    @classmethod
    def enrich_full_prompt(cls, role: str, task: str, context: str, constraints: str) -> Dict[str, str]:
        """
        Enriches all components of a prompt
        
        Returns:
            Dictionary with enriched role, task, context, and constraints
        """
        task_type = cls.detect_task_type(task)
        
        return {
            "role": cls.enrich_role(role, task),
            "task": cls.enrich_task(task),
            "context": cls.enrich_context(context, role, task),
            "constraints": cls.enrich_constraints(constraints, task_type)
        }