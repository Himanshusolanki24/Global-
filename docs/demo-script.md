# RightSize AI · demo voiceover script

Video: `RightSize-demo.mp4` · 4:51 · 1920×1080 · silent track, ready for your voice.  
Data: run `RUN-20261002-2218-c492` on 2× Tesla T4 32.2 GB + Intel(R) Xeon(R) CPU @ 2.00GHz · 30 questions per task.

**How to record:** play the video, read each block when its timestamp appears. Pace is about 2.5 words per second (150 a minute); every block fits inside its scene with a breath to spare. Numbers are spoken in words so they are easy to read aloud.

| Time | Scene | Say this |
|---|---|---|
| 0:00–0:09 | Title | This is RightSize AI. How small can a language model be before quality breaks, and what does going bigger really cost? |
| 0:09–0:25 | The problem | Data centres could use about nine hundred and forty-five terawatt-hours of electricity by 2030, yet most teams pick an AI model by habit. Nobody tells them the right size for their task, language and hardware. RightSize measures it. |
| 0:25–0:45 | How it works | Nothing runs on a laptop. One Kaggle notebook with two Tesla T4 GPUs runs every model through Ollama, measures it, and trains the router. It writes one results file, which a static site on Vercel reads. Energy comes from the GPUs' hardware counters, minus the twenty-watt idle draw. |
| 0:45–0:52 | Kaggle: the notebook | Here is that notebook on Kaggle. The API keys stay in Kaggle secrets. |
| 0:52–0:59 | Kaggle: Ollama and GPUs | Ollama installs natively, both T4s are detected, and all fifteen model tags are checked before anything downloads. |
| 0:59–1:13 | Kaggle: idle baseline and smoke test | We measure sixty seconds of idle power, twenty point three watts, run a smoke test, then the full benchmark. Every call is saved the moment it finishes, so a timeout simply resumes. |
| 1:13–1:19 | Kaggle: router and export | Finally, the notebook trains the router and exports the results. |
| 1:19–1:34 | What we measured | Six open models, from Qwen at half a billion parameters to Llama 3.1 at eight billion, on six tasks including Hindi. Two thousand and four calls in total, using under forty watt-hours above idle. |
| 1:34–1:56 | UI: Bench | This is the Bench. Turn the dial from the smallest model to the largest, and accuracy, energy, latency and cost all re-measure. Raise the floor to eighty percent, and the dial marks where quality breaks: on Q and A, the right size is Gemma 2 2B. |
| 1:56–2:14 | Finding 1 | Finding one. On Q and A, Gemma 2 2B scored eighty-three percent against eighty for Llama 8B, using three point one times less energy, and the difference is not significant. On extraction they tie, at three and a half times less energy. |
| 2:14–2:42 | UI: Frontier | The Frontier plots accuracy against energy on a log scale, with ninety-five percent confidence whiskers on both axes. The stepped line is the Pareto frontier: nothing to its left is more accurate. Switch to math, and the frontier climbs toward the larger models. And there is a dark mode. |
| 2:42–2:58 | Finding 2 | Finding two: size still pays for reasoning. On math, Llama 8B beats Qwen half-B by fifty-two points, a significant gap, for about ten times the energy. Gemma 2B reaches seventy-three percent at a fifth of the 8B model's energy. |
| 2:58–3:20 | UI: Receipt | The Receipt prices quality. Pick two models and it prints the rupees and watt-hours you pay for each extra accuracy point, and says plainly when there is no significant gain. |
| 3:20–3:38 | Finding 3 | Finding three. Quantising Phi-3.5 from FP16 to Q4 halves its energy but costs nine point three points in English, a significant drop. And Hindi is hard for every model: the best reached forty-seven percent, and the small models scored twenty-one to thirty-three. |
| 3:38–3:59 | UI: Router | The Router sends each query to the small model, unless a classifier or the small model's own self-consistency says it is unsure; then it escalates to the big one. Every ticket here is a real test query, and the tally shows the measured result. |
| 3:59–4:21 | Finding 4 | An honest negative: routing Phi-3.5 to Llama 8B kept ninety-two percent of the quality but used twenty-four more watt-hours per thousand queries. Three confidence checks cost more than one answer from a model only twice the size. Routing pays off when the big model is far costlier, like a cloud API. |
| 4:21–4:40 | UI: Method | The Method screen documents the protocol: what is metered, the idle baseline, and how every confidence interval is computed. |
| 4:40–4:51 | Close | RightSize. Measure first, then pick the smallest model that is good enough. Every number in this video was measured, with its confidence interval. Thank you. |

## Checked facts behind the script

Every figure is read from `frontend/public/results.json` (and the run database for call count and total energy).
The 945 TWh figure is the IEA *Energy and AI* (2025) projection already cited on the Bench screen; verify the citation before submitting.
