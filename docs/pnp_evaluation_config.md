# PNP evaluation configuration

These are shared evaluator defaults, not measurements from completed runs. All four benchmarks reuse PNP policy/tool interfaces. Run manifests record any explicit overrides.

Accuracy uses the final submitted answer/program, independent of process reward. Missing or invalid submissions remain in the denominator. References and hidden tests are grading-only.

Total completion budgets count generated tokens across all interaction steps; context limits also account for observations and image tokens. Tensor parallel size defaults to 1 per replica; the selected GPU count determines replica count.

```latex
% Requires \usepackage{booktabs}

\begin{table}[!ht]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Sampling temperature & 0.0 \\
Top-$p$ / Top-$k$ & 1.0 / -1 \\
Trajectories per problem $n$ & 1 \\
Evaluation metric & Integer accuracy \\
Total completion budget & 4096 tokens \\
Per-step generation cap & 820 tokens \\
Maximum interaction steps & 5 \\
Initial prompt limit & 2048 tokens \\
Context limit & 8192 tokens \\
Inference engine & vLLM \\
Mode & Offline Python API \\
Tensor parallel size & 1 per replica \\
Precision / Seed & bfloat16 / 42 \\
\bottomrule
\end{tabular}
\caption{PNP evaluation configuration for AIME2025.}
\label{tab:pnp_eval_aime2025}
\end{table}
\newpage

\begin{table}[!ht]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Sampling temperature & 0.0 \\
Top-$p$ / Top-$k$ & 1.0 / -1 \\
Trajectories per problem $n$ & 1 \\
Evaluation metric & Answer EM (F1 auxiliary) \\
Total completion budget & 4096 tokens \\
Per-step generation cap & 1024 tokens \\
Maximum interaction steps & 4 \\
Per-step prompt limit & 8192 tokens \\
Context limit & 12288 tokens \\
Inference engine & vLLM \\
Mode & Offline Python API \\
Tensor parallel size & 1 per replica \\
Precision / Seed & bfloat16 / 42 \\
\bottomrule
\end{tabular}
\caption{PNP evaluation configuration for HotpotQA.}
\label{tab:pnp_eval_hotpotqa}
\end{table}
\newpage

\begin{table}[!ht]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Sampling temperature & 0.0 \\
Top-$p$ / Top-$k$ & 1.0 / -1 \\
Trajectories per problem $n$ & 1 \\
Evaluation metric & Pass@1 \\
Total completion budget & 10240 tokens \\
Per-step generation cap & 2048 tokens \\
Maximum interaction steps & 5 \\
Per-step prompt limit & 8192 tokens \\
Context limit & 12288 tokens \\
Inference engine & vLLM \\
Mode & Offline Python API \\
Tensor parallel size & 1 per replica \\
Precision / Seed & bfloat16 / 42 \\
\bottomrule
\end{tabular}
\caption{PNP evaluation configuration for LiveCodeBench.}
\label{tab:pnp_eval_livecodebench}
\end{table}
\newpage

\begin{table}[!ht]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Sampling temperature & 0.0 \\
Top-$p$ / Top-$k$ & 1.0 / -1 \\
Trajectories per problem $n$ & 1 \\
Evaluation metric & Answer accuracy \\
Total completion budget & 6144 tokens \\
Per-step generation cap & 2048 tokens \\
Maximum interaction steps & 3 \\
Per-step prompt limit & 8192 tokens \\
Context limit & 12288 tokens \\
Inference engine & vLLM \\
Mode & Offline Python API \\
Tensor parallel size & 1 per replica \\
Precision / Seed & bfloat16 / 42 \\
\bottomrule
\end{tabular}
\caption{PNP evaluation configuration for MATH-Vision.}
\label{tab:pnp_eval_mathvision}
\end{table}
\newpage

```

Frozen dataset identity (version, revision, split, question IDs and file SHA256), checkpoint identity, code hashes, budgets and deployment settings are recorded per run.

HotpotQA uses the prepared validation split aligned with the retrieval evidence sidecar. LiveCodeBench uses release_v6; public tests may be used during interaction, and all public/private tests grade the final submission after generation. MATH-Vision accepts choice letters or option values and uses bounded numeric comparison following MATH-V's two-decimal convention. Unsupported symbolic expressions fail closed; this is not a claim of bit-for-bit equivalence with the upstream latex2sympy checker.
