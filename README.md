# Braintrust Inference Model Benchmarking

A comprehensive benchmarking framework for evaluating inference models using Braintrust. This project compares the performance of multiple AI models across different task categories and difficulty levels.

## 🎯 Models Being Tested

### Currently Active Models
- **Gemini-1.5-Pro** (Google) - `gemini-1.5-pro`
- **Gemini-2.5-Pro** (Google) - `gemini-2.5-pro`

### Available Models (Configurable)
- **GPT-5-mini** (OpenAI) - `openai:gpt-5-mini`
- **GPT-5** (OpenAI) - `openai:gpt-5`
- **GPT-4.1** (OpenAI) - `openai:gpt-4.1`
- **GPT-4.1-mini** (OpenAI) - `openai:gpt-4.1-mini`
- **Claude-3.7-Sonnet** (Anthropic) - `claude-3-7-sonnet-20250219`
- **Claude-Sonnet-4** (Anthropic) - `claude-sonnet-4-20250514`

## ✨ Features

- **Multi-model evaluation** across different AI providers (OpenAI, Anthropic, Google)
- **Comprehensive dataset** with 18 test cases across 4 categories
- **Category-based analysis** (context understanding, inference, problem context, multi-step reasoning)
- **Difficulty levels** (easy, medium, hard)
- **Factuality scoring** using structured evaluation
- **Latency measurement** with precise timing and throughput analysis
- **Cost analysis** with per-model pricing and efficiency metrics
- **Comparative analysis** with side-by-side performance comparisons
- **User experience insights** including reliability and success rates
- **Visualization** with charts and graphs for performance analysis
- **Braintrust integration** for experiment tracking and visualization
- **Flexible model configuration** with enable/disable options per model

## 📊 Dataset Categories

### Task Categories
- **Context Understanding**: Analyzing and interpreting contextual information
- **Inference**: Drawing logical conclusions from given information
- **Problem Context**: Understanding complex problem scenarios
- **Multi-step Reasoning**: Solving problems requiring multiple logical steps

### Difficulty Levels
- **Easy**: Straightforward questions with clear answers
- **Medium**: Moderately complex tasks requiring some analysis
- **Hard**: Complex problems requiring deep understanding

### Test Case Distribution
- **Context Understanding**: 6 test cases
- **Inference**: 6 test cases  
- **Problem Context**: 4 test cases
- **Multi-step Reasoning**: 2 test cases

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Braintrust account and API key
- API keys for OpenAI, Anthropic, and Google

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd braintrust-inf-model-benchmarking
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env.example .env.local
   ```
   
   Edit `.env.local` and add your API keys:
   ```env
   BRAINTRUST_API_KEY=your_braintrust_api_key
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Configure which models to test**
   ```bash
   python configure_models.py
   ```
   This interactive script lets you enable/disable specific models and providers.

6. **Configure Braintrust API keys**
   - Go to [Braintrust Dashboard](https://www.braintrust.dev/app/settings?subroute=secrets)
   - Navigate to **Settings → Organization → AI providers**
   - Add your API keys:
     - **OpenAI**: `OPENAI_API_KEY` (for GPT models)
     - **Anthropic**: `ANTHROPIC_API_KEY` (for Claude models)
     - **Google**: `GOOGLE_API_KEY` (for Gemini models)

### Usage

1. **Activate virtual environment** (if not already activated)
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Check your setup**
   ```bash
   python run_benchmark.py setup
   ```

3. **Run basic benchmark**
   ```bash
   python run_benchmark.py basic
   ```

4. **Run enhanced benchmark** (category and difficulty analysis)
   ```bash
   python run_benchmark.py enhanced
   ```

5. **Run comparative analysis** (comprehensive performance comparison)
   ```bash
   python run_benchmark.py comparative
   ```

6. **Debug API issues**
   ```bash
   python run_benchmark.py debug
   ```

7. **Run everything** (setup check + basic + enhanced + comparative)
   ```bash
   python run_benchmark.py all
   ```

### Alternative: Direct Script Execution

You can also run scripts directly:

```bash
# Activate virtual environment first
source venv/bin/activate

# Then run individual scripts
python src/setup_check.py
python src/benchmark.py
python src/enhanced_benchmark.py
python src/debug_test.py
```

### Model Discovery Tools

Check which models are available with your API keys:

```bash
# List available OpenAI models (including GPT-5)
python list_openai_models.py

# List available Anthropic models
python list_anthropic_models.py
```

## 📈 Understanding Results

### Metrics Tracked

- **Factuality Score**: How well the model's response matches expected facts
- **Latency**: Response time in seconds with precise measurement
- **Token Usage**: Input, output, and total token consumption
- **Cost Analysis**: Per-request and total cost calculations
- **Throughput**: Tokens processed per second
- **Success Rate**: Percentage of successful API calls
- **Choice Analysis**: Detailed breakdown of evaluation choices (A, B, C, D)
- **Rationale**: Reasoning behind the evaluation
- **Error Rates**: Frequency of API failures or parsing errors

### Score Interpretation

- **1.0**: Perfect factual consistency
- **0.6-0.9**: Good factual consistency with minor differences
- **0.3-0.5**: Partial factual consistency
- **0.0**: No factual consistency or contradictory information
- **-1**: Error occurred during evaluation

### Viewing Results

1. **Braintrust Dashboard**: Visit your Braintrust project to see detailed results
2. **Experiment Comparison**: Compare models across different categories and difficulties
3. **Performance Analysis**: Analyze latency, token usage, and error patterns
4. **Comparative Analysis**: View side-by-side performance comparisons with rankings
5. **Generated Files**: Check CSV files and visualizations for detailed insights
6. **Cost Analysis**: Review cost efficiency and pricing comparisons

## 🏗️ Project Structure

```
braintrust-inf-model-benchmarking/
├── src/
│   ├── benchmark.py              # Main benchmarking script with latency/cost metrics
│   ├── enhanced_benchmark.py     # Detailed category/difficulty analysis
│   ├── comparative_analysis.py   # Comprehensive performance comparison
│   ├── dataset.py                # Test cases and dataset utilities
│   ├── setup_check.py            # Setup validation script
│   └── debug_test.py             # API debugging script
├── venv/                         # Virtual environment (created during setup)
├── requirements.txt              # Python dependencies
├── run_benchmark.py              # Easy-to-use runner script
├── env.example                   # Environment variables template
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🔧 Customization

### Adding New Models

Edit `src/benchmark.py` and add to the `models` list:

```python
models = [
    # ... existing models ...
    ModelConfig("Your-Model-Name", "Provider", "model-identifier"),
]
```

### Adding New Test Cases

Edit `src/dataset.py` and add to the `benchmark_dataset` list:

```python
TestCase(
    input={
        "input": "Your question here?",
        "output": "Model's response here.",
        "expected": "Expected answer here."
    },
    expected=0.8,  # Expected score (0.0 to 1.0)
    category="your_category",
    difficulty="medium"  # easy, medium, or hard
)
```

### Modifying Evaluation Criteria

Edit the template in `src/benchmark.py`:

```python
template_yaml = """
prompt: |
  Your custom evaluation prompt here...
  
choice_scores:
  A: 0.3
  B: 0.6
  C: 1.0
  D: 0.0
"""
```

## 🐛 Troubleshooting

### Common Issues

1. **"No API keys found" error**
   - Ensure API keys are configured in Braintrust dashboard
   - Check that environment variables are set correctly
   - Run `python run_benchmark.py setup` to verify configuration

2. **Import errors**
   - Make sure virtual environment is activated: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version: `python --version` (requires 3.8+)

3. **Model not available**
   - Some models may not be available in your Braintrust account
   - Try alternative models or contact Braintrust support

4. **Evaluation errors**
   - Run `python run_benchmark.py debug` to identify specific issues
   - Check model response parsing in the evaluation function

### Virtual Environment Management

```bash
# Create new virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate virtual environment
deactivate

# Remove virtual environment (if needed)
rm -rf venv
```

### Debug Commands

```bash
# Activate virtual environment first
source venv/bin/activate

# Then run debug commands
python run_benchmark.py setup
python run_benchmark.py debug
python -u src/benchmark.py
```

## 📚 References

- [Braintrust Documentation](https://www.braintrust.dev/docs)
- [Provider Benchmark Recipe](https://www.braintrust.dev/docs/cookbook/recipes/ProviderBenchmark)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Google AI Documentation](https://ai.google.dev/docs)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project requires valid API keys and Braintrust account access. Make sure to follow the setup instructions carefully and configure your API keys in the Braintrust dashboard.
