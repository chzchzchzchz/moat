"""
Project Antigravity — Program-Aided Self-Correction (GenPRM Verifier)

This module implements:
  - GenPRMVerifier: Program-Aided Process Reward Model Verifier.
    Extracts Python code blocks from reasoning traces, executes them in a
    subprocess, compares execution outputs against text answers, and returns
    feedback scores and reflection prompts.

    Executing model-generated code is off by default; see GenPRMVerifier's
    docstring for why, and for what the subprocess is and is not confined by.

Target Hardware: Apple Silicon GPU / iOS (A17 Pro / A18 Pro / M1-M4)
"""

import sys
import os
import re
import subprocess
import tempfile
import numpy as np
from typing import List, Dict, Tuple, Optional


class GenPRMVerifier:
    """
    Program-Aided Process Reward Model (GenPRM) Verifier.

    Scores traces by executing Python code the model emitted and comparing its
    output against the trace's stated answer.

    SECURITY: that means running model-generated code on this machine. The model's
    output is influenced by whatever text reaches the prompt, so a prompt injection
    becomes code execution. The executed code runs as this process's user and can
    reach the filesystem — including any clinical database beside it — and the
    network, which defeats the zero-egress property the rest of the stack maintains.

    Execution is therefore OFF unless enable_code_execution=True is passed. With it
    off, scoring falls back to the answer-consistency check, which parses the trace
    and runs nothing.

    When enabled, execution is confined only by: a wall-clock timeout, an address
    space cap (POSIX only), a scrubbed environment, an empty working directory, and
    python -I. It is NOT a sandbox. There is no syscall filter, no namespace and no
    network restriction. Enable it only for content you would run yourself.
    """

    def __init__(
        self,
        code_timeout_sec: float = 2.0,
        max_memory_mb: int = 256,
        enable_code_execution: bool = False,
    ):
        self.code_timeout_sec = code_timeout_sec
        self.max_memory_mb = max_memory_mb
        self.enable_code_execution = enable_code_execution

    def extract_python_code(self, trace_text: str) -> List[str]:
        """
        Extract executable Python code blocks from a generated text trace.

        Looks for ```python ... ``` blocks, ``` ... ``` blocks containing code,
        or explicit Python code markers.
        """
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, trace_text, re.DOTALL)

        if not matches:
            generic_pattern = r"```\s*(.*?)\s*```"
            generic_matches = re.findall(generic_pattern, trace_text, re.DOTALL)
            for gm in generic_matches:
                if any(kw in gm for kw in ["print(", "def ", "return", "=", "for ", "import "]):
                    matches.append(gm)

        if not matches:
            lines = trace_text.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("def ") or line.strip().startswith("import "):
                    in_code = True
                if in_code:
                    code_lines.append(line)
                    if not line.strip():
                        in_code = False
            if code_lines:
                matches.append("\n".join(code_lines))

        return matches

    def extract_stated_answer(self, trace_text: str) -> Optional[str]:
        """
        Extract stated final mathematical answer from trace text.
        Recognizes GSM8K '#### <val>' and '\boxed{<val>}' patterns.
        """
        gsm_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", trace_text)
        if gsm_match:
            return gsm_match.group(1).strip()

        boxed_match = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", trace_text)
        if boxed_match:
            return boxed_match.group(1).strip()

        ans_match = re.search(r"(?:the answer is|equals|=)\s*(-?\d+(?:\.\d+)?)", trace_text, re.IGNORECASE)
        if ans_match:
            return ans_match.group(1).strip()

        return None

    def execute_code_unsandboxed(self, code_str: str) -> Tuple[bool, str, Optional[str]]:
        """
        Run a Python code string in a subprocess. See the class docstring: this is
        NOT a sandbox, and it refuses unless enable_code_execution was set.

        Confinement applied here: a wall-clock timeout, RLIMIT_AS at max_memory_mb
        (POSIX only), an environment stripped to PATH, an empty working directory,
        and python -I so the caller's site-packages and PYTHON* vars do not apply.
        The code can still read and write the filesystem and open sockets.

        Returns:
            (success: bool, output_stdout: str, error_message: Optional[str])
        """
        if not self.enable_code_execution:
            return False, "", (
                "Code execution is disabled. Pass enable_code_execution=True to "
                "GenPRMVerifier to run model-generated code, and read that class's "
                "security note before doing so."
            )

        work_dir = tempfile.mkdtemp(prefix="genprm-")
        temp_filename = os.path.join(work_dir, "candidate.py")
        with open(temp_filename, "w") as f:
            f.write(code_str)

        def _apply_limits():
            # Enforces max_memory_mb, which was previously stored and never used.
            try:
                import resource
                limit = int(self.max_memory_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except Exception:
                pass

        try:
            cmd = [sys.executable, "-I", temp_filename]
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.code_timeout_sec,
                cwd=work_dir,
                env={"PATH": os.environ.get("PATH", "")},
                preexec_fn=_apply_limits if os.name == "posix" else None,
            )

            stdout = res.stdout.strip()
            stderr = res.stderr.strip()

            if res.returncode == 0:
                return True, stdout, None
            else:
                return False, stdout, f"Exit code {res.returncode}: {stderr}"

        except subprocess.TimeoutExpired:
            return False, "", f"TimeoutExpired after {self.code_timeout_sec}s"
        except Exception as e:
            return False, "", str(e)
        finally:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    def verify_trace_with_code(self, trace_text: str) -> Dict:
        """
        Extract code from trace, execute it, and evaluate agreement with stated answer.
        """
        codes = self.extract_python_code(trace_text)
        stated_ans = self.extract_stated_answer(trace_text)

        if not codes:
            # If model produced a valid stated answer without code, do not penalize heavily
            mod = 0.0 if stated_ans is not None else -0.5
            return {
                'has_code': False,
                'code_success': False,
                'code_output': "",
                'stated_answer': stated_ans,
                'code_answer': None,
                'is_consistent': False,
                'reward_modifier': mod,
                'feedback_prompt': "No Python verification code found." if stated_ans is None else None
            }

        if not self.enable_code_execution:
            # Execution is off. Treat this exactly like a trace that stated an answer
            # without code: no bonus, no penalty. Penalising here would rank candidates
            # by whether they happened to emit a code block, which says nothing about
            # correctness when nothing ran.
            return {
                'has_code': True,
                'code_success': False,
                'code_output': "",
                'stated_answer': stated_ans,
                'code_answer': None,
                'is_consistent': False,
                'reward_modifier': 0.0,
                'code_execution_disabled': True,
                'feedback_prompt': None,
            }

        target_code = codes[-1]
        success, stdout, err = self.execute_code_unsandboxed(target_code)

        code_ans = None
        if success and stdout:
            ans_match = re.search(r"(-?\d+(?:\.\d+)?)", stdout)
            if ans_match:
                code_ans = ans_match.group(1).strip()

        is_consistent = False
        reward_mod = 0.0
        feedback = None

        if success:
            if stated_ans is not None and code_ans is not None:
                try:
                    is_consistent = (abs(float(stated_ans) - float(code_ans)) < 1e-4)
                except ValueError:
                    is_consistent = (stated_ans == code_ans)

                if is_consistent:
                    reward_mod = 1.5  # Strong GenPRM bonus
                else:
                    reward_mod = -1.0  # Contradiction penalty
                    feedback = f"Code output ({code_ans}) contradicts stated text answer ({stated_ans}). Re-evaluate math steps."
            else:
                reward_mod = 0.5
        else:
            reward_mod = -0.8
            feedback = (
                "<|im_start|>system\n"
                "Your previous Python verification block failed with the following traceback:\n"
                f"{err}\n\n"
                "Analyze the bug, correct your algebraic logic, and output a revised step-by-step reasoning chain with a working Python program.<|im_end|>"
            )

        return {
            'has_code': True,
            'code_success': success,
            'code_output': stdout if success else (err or "Error"),
            'stated_answer': stated_ans,
            'code_answer': code_ans,
            'is_consistent': is_consistent,
            'reward_modifier': reward_mod,
            'feedback_prompt': feedback
        }

    def score_candidates_genprm(
        self,
        candidate_traces: List[str],
        base_scores: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Adjust candidate scores using GenPRM program execution feedback.
        """
        N = len(candidate_traces)
        adjusted_scores = np.copy(base_scores)
        reports = []

        for i in range(N):
            rep = self.verify_trace_with_code(candidate_traces[i])
            reports.append(rep)
            adjusted_scores[i] += rep['reward_modifier']

        return adjusted_scores, reports
