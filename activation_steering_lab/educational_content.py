"""
Educational Content for Activation Steering Lab
Provides explanations, examples, and learning materials.
"""

# Main concepts explanations
EXPLANATIONS = {
    'activations_vs_embeddings': """
## 🧠 Activations vs Embeddings

**Embeddings** (Layer 0):
- Static lookup table: word → vector
- Example: "cat" → [0.2, -0.5, 0.8, ...]
- Same vector every time
- No context awareness

**Activations** (Layers 1-N):
- Dynamic, context-dependent representations
- "cat" in "I love my cat" ≠ "cat" in "the cat is on the mat"
- Built through layer-by-layer transformation
- Incorporate surrounding context

**Why we steer activations, not embeddings:**
- Activations contain high-level concepts (emotion, style, intent)
- Embeddings are too primitive
- Middle layer activations are where semantic meaning lives
""",

    'why_addition': """
## ➕ Why Addition Works

**Why add instead of replace?**

When we inject a concept vector, we use **addition**:
```
new_activation = original_activation + strength * concept_vector
```

**Why not replacement?**
- ❌ Replacement destroys context: "The weather is _____"
- ❌ Model loses track of what it's talking about
- ❌ Output becomes incoherent

**Why addition?**
- ✅ Preserves original meaning
- ✅ "Nudges" the model's thinking
- ✅ Think of it as adding a bias or preference

**Analogy:**
Replacement = Deleting your thought and replacing with a new one
Addition = Keeping your thought but viewing it through a different mood
""",

    'layer_roles': """
## 🏗️ What Happens at Each Layer?

Transformer layers progressively build up meaning:

**Early Layers (0-25%):** 🔤
- Token identification
- Syntactic parsing (subject, verb, object)
- Local patterns (n-grams)
- **Steering here affects:** Grammar, word choice

**Middle-Early (25-50%):** 💭
- Concept formation
- Semantic relationships
- Entity recognition
- **Steering here affects:** Topic, basic sentiment

**Middle-Late (50-75%):** 🎯 ← **SWEET SPOT**
- Abstract reasoning
- Emotional tone
- Writing style
- Intent understanding
- **Steering here affects:** Mood, style, personality

**Late Layers (75-100%):** 📝
- Output token selection
- Final formatting decisions
- Coherence checks
- **Steering here affects:** Specific word choices (but concepts already decided)

**Key Insight:**
Concepts "crystallize" in middle layers, then get "implemented" in late layers.
Steering middle layers = changing the plan
Steering late layers = changing the execution (less effective)
""",

    'strength_parameter': """
## 💪 Understanding Strength

The strength parameter controls **how much** we inject:
```
injection = strength × concept_vector
```

**Strength Guidelines:**

**0.5-1.0:** Subtle nudge
- Gentle influence
- Model maintains mostly original behavior
- Good for: Fine-tuning existing outputs

**1.5-2.5:** Clear effect (recommended starting point)
- Obvious steering
- Model still coherent
- Good for: Most use cases

**3.0-5.0:** Strong steering
- Dramatic changes
- May reduce coherence
- Good for: Demonstrations, experiments

**5.0+:** Overwhelming
- Concept dominates everything
- Often produces repetitive or incoherent text
- Good for: Understanding limits, breaking things

**Why higher isn't always better:**
- Too strong → model "forgets" the original task
- Optimal strength varies by concept and layer
- Experimentation is key!
""",

    'concept_quality': """
## ✨ What Makes a Good Concept?

Not all concepts work equally well for steering!

**Good Concepts:** ✅
- Clear emotional states: happy, sad, angry
- Distinct writing styles: formal, casual, poetic
- Strong personality traits: confident, timid
- Concrete tones: sarcastic, enthusiastic

**Why they work:**
- Create distinct activation patterns
- Have clear opposites/baselines
- Represented consistently across contexts

**Challenging Concepts:** ⚠️
- Abstract ideas: justice, truth, beauty
- Subtle distinctions: pleased vs. content
- Multi-dimensional: professional (formal? confident? cold?)
- Context-dependent: appropriate, normal

**Why they're harder:**
- Fuzzy activation patterns
- Overlap with many other concepts
- Require specific contexts to manifest

**Pro Tip:**
Test concepts with extreme examples:
- "happy" → "I'm so happy!" (clear)
- "appropriate" → "This is appropriate" (vague)
""",

    'baseline_importance': """
## ⚖️ The Baseline Matters

When extracting concepts, we compute:
```
concept_vector = activation(concept_prompt) - activation(baseline_prompt)
```

**Why subtract?**
The baseline removes shared/irrelevant features!

**Example:**
- Concept: "I am very happy about the weather today"
- Baseline: "I am neutral about the weather today"
- Vector: Isolates "happy" while preserving "weather" context

**Bad Baseline Example:**
- Concept: "I am very happy about the weather today"
- Baseline: "Elephants are large mammals"
- Vector: Captures "happy" + "weather" + "time" + everything else different

**Baseline Best Practices:**
1. **Match structure:** Same sentence pattern
2. **Match content:** Same topic/subject
3. **Only vary the concept:** Change one thing
4. **Use neutral language:** "neutral", "normal", "typical"

**Good Pairs:**
- "I feel happy" / "I feel neutral"
- "Write formally" / "Write normally"
- "Arrr, matey!" / "Hello, friend!"
""",

    'multi_concept': """
## 🎨 Mixing Multiple Concepts

You can inject multiple concepts simultaneously!

**Same Layer, Mixed Concepts:**
```
0.7 × happy + 0.3 × excited → cheerful
0.5 × formal + 0.5 × friendly → professional-warm
```

**Different Layers, Different Concepts:**
- Layer 15: formal (shapes overall tone)
- Layer 25: enthusiastic (adds energy)
= Professional but energetic writing

**Opposing Concepts (Concept Fighting):**
Inject contradictory concepts to see layer precedence:
- Layer 10: formal
- Layer 20: casual

**What wins?**
Usually later layers (they process the earlier layer's output)
But both influence the final result!

**Creative Combinations:**
- happy + pirate = Cheerful pirate
- sad + shakespeare = Melancholic poetry
- angry + brief = Terse frustration

**Limitations:**
- Too many concepts = confused model
- Very strong opposing concepts = incoherence
- Some combinations don't blend well
"""
}

# Example prompts for testing
EXAMPLE_PROMPTS = {
    'neutral': [
        "The weather today is",
        "In my opinion,",
        "Let me tell you about",
        "The most important thing to know is",
        "Here's what happened:"
    ],

    'storytelling': [
        "Once upon a time, there was",
        "The adventure began when",
        "Nobody expected what happened next:"
    ],

    'explanation': [
        "Quantum physics works by",
        "The reason why cats purr is",
        "To understand how cars work,",
        "The history of democracy began"
    ],

    'conversation': [
        "When someone asks me about my day,",
        "If I had to give advice,",
        "My thoughts on this topic are"
    ]
}

# Pre-defined concept pairs for extraction
CONCEPT_PAIRS = {
    'emotions': [
        ('happy', 'I feel absolutely joyful and wonderful!', 'I feel neutral today.'),
        ('sad', 'I feel deeply sorrowful and melancholic.', 'I feel neutral today.'),
        ('angry', 'I am furious and absolutely livid!', 'I feel neutral today.'),
        ('fearful', 'I am terrified and extremely afraid!', 'I feel neutral today.'),
        ('excited', 'I am SO excited and thrilled about this!', 'I feel neutral today.'),
        ('calm', 'I feel peaceful, serene, and completely at ease.', 'I feel neutral today.'),
    ],

    'styles': [
        ('formal', 'In accordance with established protocols and procedures, one must proceed accordingly.',
         'This is how you should do things.'),
        ('casual', 'Hey dude, like, just do whatever feels right, ya know?',
         'This is how you should do things.'),
        ('poetic', 'Like whispers of wind through ancient trees, the thoughts drift softly.',
         'I am thinking about something.'),
        ('technical', 'The systematic implementation of protocol-driven methodologies ensures optimal outcomes.',
         'Doing things correctly leads to good results.'),
    ],

    'personalities': [
        ('pirate', 'Arrr matey! Shiver me timbers, ye scurvy dog!', 'Hello there, friend!'),
        ('shakespeare', 'Hark! What light through yonder window breaks? \'Tis a sight most wondrous!',
         'Look at that over there!'),
        ('enthusiastic', 'This is ABSOLUTELY AMAZING! I\'m SO incredibly excited about this WONDERFUL thing!',
         'This is pretty interesting.'),
        ('pessimistic', 'Everything will probably go wrong, as it always does. There\'s no hope.',
         'Things might work out okay.'),
        ('confident', 'I am absolutely certain and completely sure of my abilities. I will succeed.',
         'I think I can probably do this.'),
    ],

    'brevity': [
        ('brief', 'Short.', 'This is a sentence that provides information in a relatively clear manner.'),
        ('verbose', 'This is a sentence that, in great detail and with numerous clauses, attempts to convey information in the most elaborate manner possible.',
         'This sentence conveys information.'),
    ]
}


def get_explanation(topic: str) -> str:
    """Get educational explanation for a topic."""
    return EXPLANATIONS.get(topic, "Explanation not found.")


def get_concept_pairs(category: str = None) -> dict:
    """
    Get concept pairs for extraction.

    Args:
        category: Specific category or None for all

    Returns:
        Dictionary of concept pairs
    """
    if category and category in CONCEPT_PAIRS:
        return {category: CONCEPT_PAIRS[category]}
    return CONCEPT_PAIRS


def get_example_prompts(category: str = None) -> list:
    """Get example prompts for testing."""
    if category and category in EXAMPLE_PROMPTS:
        return EXAMPLE_PROMPTS[category]
    # Return all prompts from all categories
    all_prompts = []
    for prompts in EXAMPLE_PROMPTS.values():
        all_prompts.extend(prompts)
    return all_prompts


def format_layer_info(layer_idx: int, total_layers: int) -> str:
    """
    Format educational information about a specific layer.

    Args:
        layer_idx: The layer index
        total_layers: Total number of layers in model

    Returns:
        Formatted string with layer information
    """
    position = layer_idx / total_layers if total_layers > 0 else 0

    if position < 0.25:
        stage = "Early"
        icon = "🔤"
        description = "Token & Syntax Processing"
        details = "These layers handle basic token meanings and grammatical structure."
        steering_effect = "Steering here affects word choice and grammar."
    elif position < 0.5:
        stage = "Middle-Early"
        icon = "💭"
        description = "Concept Formation"
        details = "These layers begin forming higher-level concepts and relationships."
        steering_effect = "Steering here affects topics and basic sentiment."
    elif position < 0.75:
        stage = "Middle-Late ⭐"
        icon = "🎯"
        description = "Abstract Reasoning (RECOMMENDED)"
        details = "These layers handle abstract concepts, emotions, and style."
        steering_effect = "Steering here has the strongest, most coherent effects!"
    else:
        stage = "Late"
        icon = "📝"
        description = "Output Decision Making"
        details = "These layers focus on choosing specific output tokens."
        steering_effect = "Steering here affects final word choices, but concepts are already decided."

    return f"""
{icon} **Layer {layer_idx}/{total_layers-1}** - {stage}

**{description}**

{details}

**Steering Effect:** {steering_effect}
    """.strip()


def generate_tutorial_steps() -> list:
    """Generate step-by-step tutorial for first-time users."""
    return [
        {
            'title': '👋 Welcome to Activation Steering!',
            'content': """
This tool lets you "steer" how a language model thinks by injecting concept vectors into its internal layers.

Think of it like adjusting the mood or style of someone's thoughts while they're thinking.
            """
        },
        {
            'title': '🧠 Step 1: Understanding Activations',
            'content': EXPLANATIONS['activations_vs_embeddings']
        },
        {
            'title': '🏗️ Step 2: How Layers Work',
            'content': EXPLANATIONS['layer_roles']
        },
        {
            'title': '➕ Step 3: Why Addition?',
            'content': EXPLANATIONS['why_addition']
        },
        {
            'title': '🎮 Step 4: Try It Yourself!',
            'content': """
Now it's your turn:

1. Go to the **Steering Playground**
2. Enter a prompt like "Tell me about the weather"
3. Select a concept like "happy"
4. Choose a middle layer (recommendation will be shown)
5. Set strength to 2.0
6. Click Generate!

Compare the normal vs steered outputs. See the difference?
            """
        }
    ]


def explain_result(normal_text: str, steered_text: str, concept: str) -> str:
    """
    Generate an explanation of the steering result.

    Args:
        normal_text: Original generation
        steered_text: Steered generation
        concept: Concept that was injected

    Returns:
        Explanation string
    """
    explanation = f"""
## Result Analysis

**Concept Injected:** {concept}

**What Changed:**
The model's internal representation was modified to include the '{concept}' concept.
This affected how it continued the text.

**Look For:**
- Different word choices reflecting {concept}
- Different tone or emotional content
- Different sentence structure or style

**Key Insight:**
The prompt is the same, but the model's "thinking" was steered in a different direction!
    """

    return explanation.strip()


def get_recommended_experiments() -> list:
    """Get list of recommended experiments for users to try."""
    return [
        {
            'name': 'Emotion Steering',
            'prompt': 'The weather today is',
            'concept': 'happy',
            'description': 'See how emotions change neutral descriptions'
        },
        {
            'name': 'Style Transfer',
            'prompt': 'Let me explain how computers work.',
            'concept': 'pirate',
            'description': 'Transform explanations into pirate speak!'
        },
        {
            'name': 'Formality Shift',
            'prompt': 'Hey, I wanted to tell you about',
            'concept': 'formal',
            'description': 'Make casual text more formal'
        },
        {
            'name': 'Brevity Control',
            'prompt': 'The most important thing to understand is',
            'concept': 'brief',
            'description': 'Force the model to be concise'
        },
        {
            'name': 'Layer Comparison',
            'prompt': 'In my opinion,',
            'concept': 'enthusiastic',
            'description': 'Try the same concept at different layers',
            'special': 'layer_analysis'
        },
        {
            'name': 'Strength Exploration',
            'prompt': 'Here is what I think:',
            'concept': 'confident',
            'description': 'Test different strength values',
            'special': 'strength_test'
        }
    ]


# Quiz questions for learning verification
QUIZ_QUESTIONS = [
    {
        'question': 'Where do abstract concepts like emotion and style primarily exist?',
        'options': [
            'In the embeddings (layer 0)',
            'In early layers (0-25%)',
            'In middle layers (50-75%)',
            'In late layers (75-100%)'
        ],
        'correct': 2,
        'explanation': 'Middle layers (50-75%) handle abstract reasoning, emotions, and style - making them ideal for steering!'
    },
    {
        'question': 'Why do we ADD concept vectors instead of REPLACING activations?',
        'options': [
            'Addition is faster to compute',
            'Addition preserves context while nudging behavior',
            'Replacement doesn\'t work at all',
            'It doesn\'t matter, both work the same'
        ],
        'correct': 1,
        'explanation': 'Addition preserves the original meaning and context while "nudging" the model\'s thinking. Replacement would destroy context.'
    },
    {
        'question': 'What does the strength parameter control?',
        'options': [
            'Which layer to inject at',
            'How many tokens to generate',
            'How much of the concept vector to inject',
            'The temperature of generation'
        ],
        'correct': 2,
        'explanation': 'Strength is a multiplier for the concept vector: injection = strength × concept_vector'
    }
]


def get_tips_and_tricks() -> list:
    """Get practical tips for effective steering."""
    return [
        "🎯 **Start in the middle**: Layers 50-75% usually work best",
        "💪 **Strength sweet spot**: Try 2.0 first, then adjust",
        "⚖️ **Match your baseline**: Keep structure the same, only change the concept",
        "🔬 **Experiment systematically**: Change one variable at a time",
        "📊 **Compare outputs**: Always generate with and without steering",
        "🎨 **Mix concepts carefully**: Too many concepts = confusion",
        "📝 **Clear concepts work best**: Strong emotions and distinct styles",
        "🔄 **Different prompts, same concept**: Test generalization",
        "⚡ **Very high strength**: Often makes output repetitive or incoherent",
        "🧪 **Opposing concepts**: Try injecting contradictory ideas at different layers!"
    ]
