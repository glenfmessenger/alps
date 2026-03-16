"""
Curated prompt set for cost prediction experiments.

Goal: Prompts that naturally elicit varied-length responses WITHOUT hitting max_tokens.

Categories:
1. SHORT (20-100 tokens): Factual, yes/no, definitions
2. MEDIUM (100-500 tokens): Explanations, comparisons, summaries  
3. LONG (500-1500 tokens): Detailed explanations, multi-part, creative

Each prompt is tagged with expected length category for analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class CuratedPrompt:
    text: str
    category: str  # "short", "medium", "long"
    subcategory: str  # more specific type


# =============================================================================
# SHORT PROMPTS (target: 20-100 tokens)
# =============================================================================

SHORT_FACTUAL = [
    "What year was Google founded?",
    "What is the capital of Australia?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How many planets are in our solar system?",
    "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?",
    "What is the speed of light in km/s?",
    "What year did World War 2 end?",
    "What is the tallest mountain in the world?",
    "Who invented the telephone?",
    "What is the smallest country in the world?",
    "What language has the most native speakers?",
    "What is the currency of Japan?",
    "Who was the first person to walk on the moon?",
    "What is the largest planet in our solar system?",
    "What year was the iPhone first released?",
    "What is the boiling point of water in Celsius?",
    "Who founded Microsoft?",
    "What is the longest river in the world?",
]

SHORT_YES_NO = [
    "Is Python an interpreted language?",
    "Is the sun a star?",
    "Is water H2O?",
    "Is Tokyo the capital of Japan?",
    "Is gold heavier than silver?",
    "Is Linux an operating system?",
    "Is the Great Wall of China visible from space?",
    "Is a tomato a fruit?",
    "Is Mount Everest in Nepal?",
    "Is JavaScript the same as Java?",
    "Is the Earth older than the Moon?",
    "Is carbon dioxide a greenhouse gas?",
    "Is Bitcoin a cryptocurrency?",
    "Is Antarctica a continent?",
    "Is light faster than sound?",
]

SHORT_DEFINITIONS = [
    "Define entropy in one sentence.",
    "What is a prime number?",
    "Define photosynthesis briefly.",
    "What is an algorithm?",
    "Define democracy in simple terms.",
    "What is a black hole?",
    "Define inflation in economics.",
    "What is DNA?",
    "Define machine learning in one sentence.",
    "What is a metaphor?",
    "Define gravity briefly.",
    "What is an ecosystem?",
    "Define capitalism in simple terms.",
    "What is a virus in biology?",
    "Define recursion in programming.",
]

SHORT_SIMPLE_QUESTIONS = [
    "Why is the sky blue? Answer briefly.",
    "What causes rain? One paragraph max.",
    "Why do leaves change color in fall? Brief answer.",
    "What makes ice float on water? Short explanation.",
    "Why do we have seasons? Brief answer.",
    "What causes thunder? One sentence.",
    "Why is the ocean salty? Brief explanation.",
    "What makes a rainbow? Short answer.",
    "Why do stars twinkle? Brief response.",
    "What causes tides? One paragraph.",
]


# =============================================================================
# MEDIUM PROMPTS (target: 100-500 tokens)
# =============================================================================

MEDIUM_EXPLANATIONS = [
    "How does photosynthesis work?",
    "Explain how a refrigerator works.",
    "How do vaccines work?",
    "Explain the water cycle.",
    "How does GPS determine your location?",
    "Explain how airplanes fly.",
    "How does the internet work?",
    "Explain how muscles grow after exercise.",
    "How do search engines rank results?",
    "Explain how solar panels generate electricity.",
    "How does the human immune system work?",
    "Explain how a car engine works.",
    "How do cryptocurrencies work?",
    "Explain how 3D printers work.",
    "How does memory work in the human brain?",
    "Explain how touchscreens detect touch.",
    "How do noise-canceling headphones work?",
    "Explain how MRI machines work.",
    "How does the stock market work?",
    "Explain how batteries store energy.",
]

MEDIUM_COMPARISONS = [
    "Compare TCP and UDP protocols.",
    "What are the differences between Python and JavaScript?",
    "Compare democracy and authoritarianism.",
    "What's the difference between weather and climate?",
    "Compare renewable and non-renewable energy.",
    "What are the differences between RAM and ROM?",
    "Compare machine learning and traditional programming.",
    "What's the difference between a virus and bacteria?",
    "Compare capitalism and socialism.",
    "What are the differences between HTTP and HTTPS?",
    "Compare electric and gasoline cars.",
    "What's the difference between a meteor and an asteroid?",
    "Compare aerobic and anaerobic exercise.",
    "What are the differences between iOS and Android?",
    "Compare classical and quantum computing.",
]

MEDIUM_SUMMARIES = [
    "Summarize the plot of Romeo and Juliet.",
    "Give a brief overview of World War 1.",
    "Summarize the theory of evolution.",
    "Give a brief overview of the French Revolution.",
    "Summarize how the internet was created.",
    "Give a brief overview of climate change.",
    "Summarize the story of the Odyssey.",
    "Give a brief overview of the Renaissance.",
    "Summarize the main ideas of Buddhism.",
    "Give a brief overview of the Cold War.",
    "Summarize how democracy developed in ancient Greece.",
    "Give a brief overview of the Industrial Revolution.",
    "Summarize the plot of 1984 by George Orwell.",
    "Give a brief overview of the space race.",
    "Summarize the main principles of economics.",
]

MEDIUM_HOW_TO = [
    "How do you make bread from scratch?",
    "Explain how to change a car tire.",
    "How do you start learning to code?",
    "Explain how to write a good resume.",
    "How do you train for a marathon?",
    "Explain how to start a small business.",
    "How do you learn a new language effectively?",
    "Explain how to give a good presentation.",
    "How do you build good habits?",
    "Explain how to negotiate a salary.",
    "How do you improve your writing skills?",
    "Explain how to manage personal finances.",
    "How do you prepare for a job interview?",
    "Explain how to meditate for beginners.",
    "How do you take better photographs?",
]


# =============================================================================
# LONG PROMPTS (target: 500-1500 tokens)
# =============================================================================

LONG_DETAILED = [
    "Explain how neural networks learn, including forward propagation, backpropagation, and gradient descent.",
    "Describe the process of evolution by natural selection with examples.",
    "Explain the causes, progression, and resolution of the American Civil War.",
    "Describe how the human digestive system works from eating to excretion.",
    "Explain the history and impact of the printing press on society.",
    "Describe the structure and function of the human heart and circulatory system.",
    "Explain how compilers translate code from high-level languages to machine code.",
    "Describe the causes and effects of the Great Depression.",
    "Explain the science behind climate change including greenhouse gases and feedback loops.",
    "Describe the life cycle of a star from formation to death.",
    "Explain the history and development of the internet from ARPANET to today.",
    "Describe how the human eye works including optics and neural processing.",
    "Explain the principles of object-oriented programming with examples.",
    "Describe the causes and consequences of the fall of the Roman Empire.",
    "Explain how CRISPR gene editing works and its potential applications.",
]

LONG_MULTIPART = [
    "What are the main causes of poverty, its effects on society, and potential solutions?",
    "Discuss the benefits, risks, and ethical considerations of artificial intelligence.",
    "Explain the causes of World War 1, key events, and its lasting consequences.",
    "What are the types of renewable energy, their advantages, disadvantages, and future potential?",
    "Discuss the history of civil rights movements, key figures, and remaining challenges.",
    "Explain what blockchain is, how it works, and its applications beyond cryptocurrency.",
    "What are the causes of mental health issues, their impact, and treatment approaches?",
    "Discuss the history of space exploration, major achievements, and future goals.",
    "Explain the scientific method, its steps, and why it's important for research.",
    "What are the effects of social media on society, both positive and negative?",
    "Discuss the history of democracy, different forms of government, and their tradeoffs.",
    "Explain how the economy works including supply and demand, inflation, and monetary policy.",
    "What are the major world religions, their core beliefs, and how they compare?",
    "Discuss the causes of environmental pollution, its effects, and solutions.",
    "Explain the history of computing from early machines to modern computers.",
]

LONG_ANALYTICAL = [
    "Analyze the pros and cons of remote work for employees and employers.",
    "Discuss the arguments for and against universal basic income.",
    "Analyze the impact of automation on employment and the economy.",
    "Discuss the ethical implications of genetic engineering in humans.",
    "Analyze the effectiveness of different education systems around the world.",
    "Discuss the arguments for and against nuclear energy.",
    "Analyze the impact of globalization on developing countries.",
    "Discuss the pros and cons of social media regulation.",
    "Analyze the effectiveness of different approaches to healthcare systems.",
    "Discuss the arguments for and against space colonization.",
    "Analyze the impact of streaming services on the entertainment industry.",
    "Discuss the ethics of animal testing in medical research.",
    "Analyze the effectiveness of international climate agreements.",
    "Discuss the pros and cons of standardized testing in education.",
    "Analyze the impact of e-commerce on traditional retail.",
]

LONG_CREATIVE = [
    "Write a short story about a scientist who discovers time travel.",
    "Create a dialogue between Socrates and a modern AI researcher.",
    "Write a short story about the last human on Earth.",
    "Create a conversation between Einstein and Newton about physics.",
    "Write a short story about a robot learning to feel emotions.",
    "Create a dialogue between a climate scientist and a skeptic.",
    "Write a short story about first contact with aliens.",
    "Create a conversation between a medieval knight and a modern soldier.",
    "Write a short story about a world without electricity.",
    "Create a dialogue between Aristotle and a modern philosopher.",
]


def get_all_prompts() -> List[CuratedPrompt]:
    """Get all curated prompts with their categories."""
    prompts = []
    
    # Short prompts
    for text in SHORT_FACTUAL:
        prompts.append(CuratedPrompt(text, "short", "factual"))
    for text in SHORT_YES_NO:
        prompts.append(CuratedPrompt(text, "short", "yes_no"))
    for text in SHORT_DEFINITIONS:
        prompts.append(CuratedPrompt(text, "short", "definition"))
    for text in SHORT_SIMPLE_QUESTIONS:
        prompts.append(CuratedPrompt(text, "short", "simple_question"))
    
    # Medium prompts
    for text in MEDIUM_EXPLANATIONS:
        prompts.append(CuratedPrompt(text, "medium", "explanation"))
    for text in MEDIUM_COMPARISONS:
        prompts.append(CuratedPrompt(text, "medium", "comparison"))
    for text in MEDIUM_SUMMARIES:
        prompts.append(CuratedPrompt(text, "medium", "summary"))
    for text in MEDIUM_HOW_TO:
        prompts.append(CuratedPrompt(text, "medium", "how_to"))
    
    # Long prompts
    for text in LONG_DETAILED:
        prompts.append(CuratedPrompt(text, "long", "detailed"))
    for text in LONG_MULTIPART:
        prompts.append(CuratedPrompt(text, "long", "multipart"))
    for text in LONG_ANALYTICAL:
        prompts.append(CuratedPrompt(text, "long", "analytical"))
    for text in LONG_CREATIVE:
        prompts.append(CuratedPrompt(text, "long", "creative"))
    
    return prompts


def get_balanced_sample(n: int, seed: int = 42) -> List[CuratedPrompt]:
    """
    Get a balanced sample across categories.
    
    Aims for roughly equal split: 33% short, 33% medium, 33% long
    """
    random.seed(seed)
    all_prompts = get_all_prompts()
    
    short = [p for p in all_prompts if p.category == "short"]
    medium = [p for p in all_prompts if p.category == "medium"]
    long = [p for p in all_prompts if p.category == "long"]
    
    n_per_category = n // 3
    remainder = n % 3
    
    random.shuffle(short)
    random.shuffle(medium)
    random.shuffle(long)
    
    # Take from each category, cycling if needed
    selected = []
    
    for i in range(n_per_category + (1 if remainder > 0 else 0)):
        if i < len(short):
            selected.append(short[i])
        else:
            selected.append(short[i % len(short)])
    
    for i in range(n_per_category + (1 if remainder > 1 else 0)):
        if i < len(medium):
            selected.append(medium[i])
        else:
            selected.append(medium[i % len(medium)])
    
    for i in range(n_per_category):
        if i < len(long):
            selected.append(long[i])
        else:
            selected.append(long[i % len(long)])
    
    random.shuffle(selected)
    return selected[:n]


def print_stats():
    """Print statistics about the prompt set."""
    all_prompts = get_all_prompts()
    
    print(f"Total prompts: {len(all_prompts)}")
    print()
    
    for category in ["short", "medium", "long"]:
        cat_prompts = [p for p in all_prompts if p.category == category]
        print(f"{category.upper()}: {len(cat_prompts)} prompts")
        
        subcats = {}
        for p in cat_prompts:
            subcats[p.subcategory] = subcats.get(p.subcategory, 0) + 1
        
        for subcat, count in sorted(subcats.items()):
            print(f"  - {subcat}: {count}")
        print()


if __name__ == "__main__":
    print_stats()
    
    print("\nSample of 10 balanced prompts:")
    print("-" * 50)
    for p in get_balanced_sample(10):
        print(f"[{p.category}] {p.text[:60]}...")
