import contextlib
import csv
import http.server
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from react_gpt.cli import main, select_models
from react_gpt.config import Config
from react_gpt.corpus import chunks, load_corpus
from react_gpt.models import ModelClient, parse_response
from react_gpt.pipeline import majority, run, rank_actionables
from react_gpt.prompts import CATEGORIES, MEDICINE_CATEGORIES
from react_gpt.reconcile import reconcile
from react_gpt.similarity import Similarity, calibrate, complete_linkage
from react_gpt.storage import read_json, RunLock


class Server(http.server.BaseHTTPRequestHandler):
    calls = []
    malformed = False
    installed = []
    pulls = []
    pull_error = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"models": [{"name": m} for m in Server.installed]}).encode())

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/pull":
            Server.pulls.append(data["model"])
            self.send_response(200)
            self.end_headers()
            if Server.pull_error:
                self.wfile.write((json.dumps({"error": Server.pull_error}) + "\n").encode())
                return
            for event in [{"status": "pulling manifest"}, {"status": "pulling layer", "total": 100, "completed": 50}, {"status": "success"}]:
                self.wfile.write((json.dumps(event) + "\n").encode())
            Server.installed.append(data["model"])
            return
        Server.calls.append(data)
        prompt = data["prompt"]
        if '"evidence_in_source"' in prompt:
            result = {"evidence_in_source": "YES", "follows_from_evidence": "YES", "category_fit": "NO",
                      "duplicate_of": "NONE", "rationale": "Source states it."}
        elif 'Derive a useful taxonomy' in prompt:
            result = {"categories": [{"name": "Mentorship", "criteria": "Direct support for new contributors"}, {"name": "Retention", "criteria": "Actions that retain contributors"}]}
        elif '"verdict"' in prompt:
            result = {"verdict": "YES" if data["model"] != "family:b" else "NO"}
        elif '"categories"' in prompt:
            result = {"categories": ["Mentorship", "Retention"] if "Mentorship" in prompt else [list(MEDICINE_CATEGORIES if "medicine categories" in prompt else CATEGORIES)[0]]}
        elif '"features"' in prompt:
            result = {"features": ["s_net_overlap"]}
        else:
            result = {"recommendations": [{"recommendation": "Provide mentorship for new contributors.",
                       "positive_impact": "Improves retention.", "evidence": "The study found improved retention.",
                       "confidence": 0.9}]}
        raw = "not JSON" if Server.malformed else json.dumps(result)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"done": True, "response": raw}).encode())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.corpus = self.base / "papers"
        self.corpus.mkdir()
        (self.corpus / "article.md").write_text("# A study\nProvide mentorship for new contributors. It improves retention.")
        Server.calls = []
        Server.malformed = False
        Server.installed = ["family:a", "family:b", "judge:c"]
        Server.pulls = []
        Server.pull_error = None
        self.config = Config(str(self.corpus), str(self.base / "output"), ["family:a", "family:b"],
                             evaluators=["family:a", "family:b", "judge:c"], similarity="lexical", threshold=0.8,
                             ollama_url=f"http://127.0.0.1:{self.server.server_port}", retries=0, features=True)

    def test_end_to_end_http_and_resume(self):
        result = run(self.config)
        self.assertEqual(result["status"], "complete")
        final = read_json(self.base / "output/final_set.json")
        self.assertEqual(len(final), 1)
        row = final[0]
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["quality_tier"], "Sound and precise")
        self.assertEqual(row["support"], 2)
        self.assertEqual(row["support_fraction"], 1)
        self.assertEqual(row["SOUND"], "YES")
        self.assertEqual(row["PRECISE"], "YES")
        self.assertEqual(row["FEATURES"], "s_net_overlap")
        self.assertEqual(len(row["quality_votes"]), 3)
        self.assertEqual(len(row["members"]), 2)
        count = len(Server.calls)
        self.assertEqual(count, 14)
        run(self.config)
        self.assertEqual(len(Server.calls), count, "resuming must not repeat model calls")
        with (self.base / "output/final_set.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        self.assertIn("Provide mentorship", (self.base / "output/report.html").read_text())

    def test_progress_and_report_source_metadata(self):
        capture = io.StringIO()
        with contextlib.redirect_stderr(capture):
            result = run(self.config)
        for label in ("Pipeline", "Read corpus", "Extract", "Score", "Categorize", "Features", "100%"):
            self.assertIn(label, capture.getvalue())
        report = (self.base / "output/report.html").read_text()
        self.assertIn((self.corpus / "article.md").resolve().as_uri(), report)
        self.assertIn("Read extracted article text", report)
        self.assertIn("It improves retention.", report)
        self.assertIn(result["last_run_started_at"][:10], report)
        self.assertIn("+00:00", result["completed_at"])
        resumed = run(self.config)
        self.assertEqual(result["started_at"], resumed["started_at"])

    def test_empty_results_keep_source_and_escape_html(self):
        class Empty:
            def preflight(self, models): pass
            def ask(self, *args): return []
        (self.corpus / "article.md").write_text('# <script>alert(1)</script>\nSource text <unsafe>.')
        run(self.config, client=Empty())
        report = (self.base / "output/report.html").read_text()
        self.assertIn("Base articles", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertIn("Source text &lt;unsafe&gt;", report)
        self.assertNotIn("<script>alert(1)</script>", report)

    def test_medicine_domain_end_to_end(self):
        self.config.domain = "medicine"
        self.config.features = False
        run(self.config)
        rows = read_json(self.base / "output/final_set.json")
        self.assertEqual(rows[0]["CATEGORY"], "Imaging and Diagnosis")
        self.assertTrue(any("medical and medical-imaging research" in c["prompt"] for c in Server.calls))
        self.assertIn("not clinical validation", (self.base / "output/report.html").read_text())
        with self.assertRaises(ValueError):
            parse_response('{"categories":["Documentation Practices"]}', "categories", MEDICINE_CATEGORIES)
        self.config.features = True
        with self.assertRaisesRegex(ValueError, "only available"):
            self.config.validate()

    def test_runtime_categories_multilabel_and_resume(self):
        self.config.category_mode = "dynamic"
        run(self.config)
        output = self.base / "output"
        taxonomy = read_json(output / "taxonomy.json")
        discovery = next(call for call in Server.calls if "Derive a useful taxonomy" in call["prompt"])
        self.assertEqual(discovery["format"]["properties"]["categories"]["maxItems"], 12)
        self.assertEqual(discovery["format"]["properties"]["categories"]["items"]["required"], ["name", "criteria"])
        verdict = next(call for call in Server.calls if '{"verdict": "YES"}' in call["prompt"])
        self.assertEqual(verdict["format"]["properties"]["verdict"]["enum"], ["YES", "NO"])
        self.assertEqual(set(taxonomy["categories"]), {"Mentorship", "Retention"})
        final = read_json(output / "final_set.json")
        self.assertEqual(final[0]["categories"], ["Mentorship", "Retention"])
        self.assertEqual(final[0]["CATEGORY"], "Mentorship | Retention")
        count = len(Server.calls)
        run(self.config)
        self.assertEqual(len(Server.calls), count)
        self.assertEqual(taxonomy, read_json(output / "taxonomy.json"))

    def test_dynamic_empty_and_batched_taxonomy(self):
        from react_gpt.taxonomy import discover_taxonomy
        self.config.category_mode = "dynamic"
        class TaxonomyClient:
            calls = []
            def ask(self, model, prompt, kind):
                self.calls.append(prompt)
                return {"Topic": "Related actions"}
        client = TaxonomyClient()
        self.assertEqual(discover_taxonomy([], self.config, client)["categories"], {})
        self.assertEqual(client.calls, [])
        rows = [{"cluster_id": str(i), "actionable": "x" * 6500} for i in range(3)]
        result = discover_taxonomy(rows, self.config, client)
        self.assertEqual(result["action_ids"], ["0", "1", "2"])
        self.assertEqual(len(result["proposals"]), 3)
        self.assertEqual(len(client.calls), 5)

    def test_taxonomy_validation(self):
        for categories in ([], [{"name": "X", "criteria": ""}],
                           [{"name": "X", "criteria": "one"}, {"name": "x", "criteria": "two"}],
                           [{"name": "a|b", "criteria": "two"}]):
            with self.assertRaises(ValueError):
                parse_response(json.dumps({"categories": categories}), "taxonomy")
        with self.assertRaises(ValueError):
            parse_response('{"categories":["unknown"]}', "categories", {"Known": "criteria"})

    def test_report_command_rebuilds_without_inference(self):
        run(self.config)
        count = len(Server.calls)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["report", str(self.base / "output")])
        self.assertEqual(len(Server.calls), count)
        report = (self.base / "output/report.html").read_text()
        self.assertIn('id="graph"', report)
        self.assertIn('id="report-data"', report)
        self.assertNotIn('src="https://', report)
        self.assertNotIn('__DATA__', report)

    def test_graph_payload_cannot_close_script_element(self):
        from react_gpt.report import render_graph
        target = self.base / "unsafe.html"
        render_graph(target, [], [{"article_title": "</script><script>alert(1)</script>__STYLE__"}],
                     {"mode": "dynamic", "categories": {}}, "medicine", None, "", "")
        text = target.read_text()
        self.assertNotIn('</script><script>alert(1)', text)
        self.assertIn('\\u003c/script\\u003e', text)
        self.assertIn('__STYLE__', text)

    def test_explicit_source_metadata_and_fingerprint(self):
        sidecar = self.corpus / "article.md.metadata.json"
        sidecar.write_text(json.dumps({"title": "Verified title", "link": "https://example.org/source"}))
        article = load_corpus(self.config)[0]
        self.assertEqual(article["article_title"], "Verified title")
        self.assertTrue(article["title_explicit"])
        run(self.config, until="preprocess")
        sidecar.write_text(json.dumps({"title": "Changed metadata"}))
        with self.assertRaisesRegex(ValueError, "different corpus/configuration"):
            run(self.config)
        sidecar.write_text(json.dumps({"title": 3}))
        with self.assertRaises(ValueError):
            load_corpus(self.config)

    def test_stage_resume(self):
        run(self.config, until="extract")
        self.assertEqual(len(Server.calls), 2)
        run(self.config)
        self.assertEqual(len(Server.calls), 14)

    def test_malformed_is_failure_not_no_action(self):
        Server.malformed = True
        with self.assertRaisesRegex(RuntimeError, "extract failed"):
            run(self.config)
        self.assertEqual(read_json(self.base / "output/manifest.json")["status"], "failed")
        self.assertFalse((self.base / "output/final_set.csv").exists())
        Server.malformed = False
        run(self.config)
        self.assertEqual(read_json(self.base / "output/manifest.json")["status"], "complete")

    def test_configuration_and_corpus_changes_rejected(self):
        run(self.config, "preprocess")
        self.config.threshold = 0.5
        with self.assertRaisesRegex(ValueError, "different corpus/configuration"):
            run(self.config)
        self.config.threshold = 0.8
        (self.corpus / "article.md").write_text("Changed article")
        with self.assertRaisesRegex(ValueError, "different corpus/configuration"):
            run(self.config)

    def test_identical_titles_do_not_collide(self):
        (self.corpus / "second.md").write_text("# A study\nOther evidence.")
        articles = load_corpus(self.config)
        self.assertEqual(len({a["article_id"] for a in articles}), 2)
        run(self.config)
        rows = read_json(self.base / "output/final_set.json")
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["cluster_id"], rows[1]["cluster_id"])

    def test_no_recommendations_exports_headers(self):
        class Empty:
            def preflight(self, models): pass
            def ask(self, *args): return []
        self.config.similarity = "hybrid"  # no embedding installation needed for empty corpus results
        run(self.config, client=Empty())
        self.assertEqual(read_json(self.base / "output/final_set.json"), [])
        self.assertIn("article_id", (self.base / "output/final_set.csv").read_text())

    def test_missing_model_fails_before_inference(self):
        self.config.models = ["missing:model"]
        with self.assertRaisesRegex(RuntimeError, "Models missing"):
            run(self.config)
        self.assertEqual(Server.calls, [])

    def test_dry_run_no_writes_or_requests(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["run", "--corpus", str(self.corpus), "--output", str(self.base / "dry"), "--models", "custom:tag", "--dry-run"])
        self.assertFalse((self.base / "dry").exists())
        self.assertEqual(Server.calls, [])
        self.assertEqual(json.loads(capture.getvalue())["config"]["models"], ["custom:tag"])

    def test_configuration_relative_to_file(self):
        config_path = self.base / "run.json"
        config_path.write_text(json.dumps({"corpus": "papers", "output": "relative-out", "models": ["a"]}))
        with contextlib.redirect_stdout(io.StringIO()):
            main(["run", "--config", str(config_path), "--until", "preprocess"])
        self.assertTrue((self.base / "relative-out/preprocess.json").exists())

    def test_wizard(self):
        answers = [str(self.corpus), str(self.base / "wizard-out"), self.config.ollama_url, "1,2", "", "anthropic:claude-opus-5-5", "", "CoT", "lexical", "", "0.8", "yes"]
        with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            main(["init", "--config", str(self.base / "wizard.json")])
        saved = read_json(self.base / "wizard.json")
        self.assertTrue(saved["features"])
        self.assertEqual(saved["evaluators"], ["family:a", "family:b"])
        self.assertEqual(saved["judge"], "anthropic:claude-opus-5-5")
        self.assertFalse(saved["think"])

    def test_judge_annotates_without_filtering(self):
        self.config.judge = "judge:c"
        run(self.config)
        final = read_json(self.base / "output/final_set.json")
        self.assertEqual(len(final), 1)
        judge = final[0]["judge"]
        self.assertEqual((judge["evidence_in_source"], judge["category_fit"], judge["duplicate_of"]), ("YES", "NO", None))
        self.assertEqual(judge["model"], "judge:c")
        call = next(c for c in Server.calls if '"evidence_in_source"' in c["prompt"])
        self.assertIn("Provide mentorship for new contributors. It improves retention.", call["prompt"])
        self.assertEqual(call["format"]["properties"]["duplicate_of"]["enum"], ["NONE"])
        self.assertIn("JUDGE_EVIDENCE", (self.base / "output/final_set.csv").read_text())
        self.assertIn("Final judge", (self.base / "output/report.html").read_text())
        self.assertFalse(any(c["think"] for c in Server.calls))

    def test_providers_and_jev_contract(self):
        from react_gpt.models import split_model, jev_questions, jev_answer
        self.assertEqual(split_model("qwen3:8b"), ("ollama", "qwen3:8b"))
        self.assertEqual(split_model("anthropic:claude-opus-5-5"), ("anthropic", "claude-opus-5-5"))
        self.assertEqual(split_model("ollama:qwen3:8b"), ("ollama", "qwen3:8b"))
        with self.assertRaisesRegex(ValueError, "only evaluate or judge"):
            Config(str(self.corpus), str(self.base / "o"), ["jev:jev-1.13.0"]).validate()
        with self.assertRaisesRegex(ValueError, "only evaluate or judge"):
            ModelClient(self.config).ask("jev:jev-1.13.0", "text", "extract")
        names = ["Mentorship", "Retention"]
        questions = jev_questions("categories", names, None)
        reply = {"answers": {"q0": {"choice": "YES"}, "q1": {"choice": "NO"}}}
        self.assertEqual(jev_answer(reply, "categories", questions, names), {"categories": ["Mentorship"]})
        with self.assertRaises(ValueError):
            jev_answer({"answers": {"q0": {"choice": "MAYBE"}, "q1": {"choice": "NO"}}}, "categories", questions, names)

    def test_remote_runs_need_confirmation(self):
        from react_gpt.cli import confirm_remote
        self.config.evaluators = ["family:a", "openai:gpt-5"]
        self.config.validate()
        with patch("sys.stdin.isatty", return_value=False), contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaisesRegex(ValueError, "--yes"):
                confirm_remote(self.config)
            confirm_remote(self.config, yes=True)
        self.assertIn("leaves this computer", out.getvalue())
        self.config.evaluators = ["family:a"]
        confirm_remote(self.config)  # Local-only runs never prompt.

    def test_pdf_title_prefers_largest_font(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("pymupdf not installed")
        from react_gpt.corpus import pdf_title
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 60), "European Journal of Radiology", fontsize=9)
        page.insert_text((72, 120), "Deep Learning for Angiography", fontsize=20)
        page.insert_text((72, 160), "Body text " * 20, fontsize=10)
        self.assertEqual(pdf_title(document), "Deep Learning for Angiography")
        document.set_metadata({"title": "A Metadata Title"})
        self.assertEqual(pdf_title(document), "A Metadata Title")

    def test_model_selector_installs_and_deduplicates(self):
        output = io.StringIO()
        with patch("builtins.input", return_value="1,new-model,new-model:latest,1"), contextlib.redirect_stdout(output):
            models = select_models(ModelClient(self.config), "Extraction")
        self.assertEqual(models, ["family:a", "new-model:latest"])
        self.assertEqual(Server.pulls, ["new-model:latest"])
        self.assertIn("1. family:a", output.getvalue())
        self.assertIn("50%", output.getvalue())

    def test_model_selector_retries_bad_number_without_download(self):
        with patch("builtins.input", side_effect=["99", "1"]), contextlib.redirect_stdout(io.StringIO()):
            models = select_models(ModelClient(self.config), "Extraction")
        self.assertEqual(models, ["family:a"])
        self.assertEqual(Server.pulls, [])

    def test_model_selector_download_error_allows_reselection(self):
        Server.pull_error = "model not found"
        output = io.StringIO()
        with patch("builtins.input", side_effect=["invalid:model", "2"]), contextlib.redirect_stdout(output):
            models = select_models(ModelClient(self.config), "Extraction")
        self.assertEqual(models, ["family:b"])
        self.assertIn("model not found", output.getvalue())

    def test_model_selector_empty_installation(self):
        Server.installed = []
        with patch("builtins.input", side_effect=["", "new-model"]), contextlib.redirect_stdout(io.StringIO()):
            models = select_models(ModelClient(self.config), "Extraction")
        self.assertEqual(models, ["new-model:latest"])
        self.assertEqual(Server.pulls, ["new-model:latest"])

    def test_evaluation_selector_installs_separate_model(self):
        with patch("builtins.input", return_value="judge:new"), contextlib.redirect_stdout(io.StringIO()):
            models = select_models(ModelClient(self.config), "Evaluation", default=["family:a"])
        self.assertEqual(models, ["judge:new"])
        self.assertEqual(Server.pulls, ["judge:new"])

    def test_validation(self):
        for field, value in [("models", ["a", "a"]), ("threshold", float("nan")), ("min_support", 3),
                             ("limit", -1), ("num_ctx", 0), ("features", "false"), ("timeout", 0)]:
            with self.subTest(field=field):
                values = self.config.to_dict()
                values[field] = value
                with self.assertRaises(ValueError): Config(**values).validate()
        self.config.output = str(self.corpus / "output")
        with self.assertRaisesRegex(ValueError, "outside"): self.config.validate()

    def test_csv_jsonl_and_legacy_envelope(self):
        file = self.base / "corpus.csv"
        with file.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Title", "Text", "Link"])
            writer.writeheader()
            writer.writerow({"Title": "Title", "Text": "Text with\nnewlines", "Link": "https://example.com"})
        self.config.corpus = str(file)
        self.assertEqual(load_corpus(self.config)[0]["text"], "Text with\nnewlines")
        file = self.base / "corpus.jsonl"
        file.write_text(json.dumps({"title": "Title", "text": "Article"}) + "\n")
        self.config.corpus = str(file)
        self.assertEqual(load_corpus(self.config)[0]["text"], "Article")
        file = self.base / "legacy.md"
        file.write_text("# Title\n\n**Source:** https://example.com\n**Original file:** /old/path.pdf\n\n---\n\nArticle")
        self.config.corpus = str(file)
        self.assertEqual(load_corpus(self.config)[0]["text"], "Article")

    def test_calibration(self):
        path = self.base / "benchmark.csv"
        path.write_text("sentence1,sentence2,label\nmentor contributors,mentor contributors,1\nmentor contributors,automate testing,0\n")
        result = calibrate(path, Similarity("lexical"))
        self.assertEqual(result["source"], "benchmark")
        self.assertGreater(result["threshold"], 0)
        self.config.benchmark = str(path)
        run(self.config)
        self.assertEqual(read_json(self.base / "output/calibration.json")["source"], "benchmark")

    def test_run_lock(self):
        out = Path(self.config.output)
        out.mkdir()
        with RunLock(out / ".lock"):
            with self.assertRaisesRegex(RuntimeError, "lock run directory"):
                run(self.config)
        run(self.config, "preprocess")

    def test_support_penalty_and_representative(self):
        article = load_corpus(self.config)[0]
        pool = []
        for i, model in enumerate(["family:a", "family:a", "family:b"]):
            pool.append(dict(article_id=article["article_id"], model=model, candidate_id=str(i),
                             recommendation=str(i), positive_impact="impact", evidence="evidence", confidence=.8))
        class Metric:
            def matrix(self, texts): return [[1,.99,.95],[.99,1,.9],[.95,.9,1]]
        rows = reconcile([article], pool, self.config, Metric(), .8)
        self.assertEqual(sorted(r["support"] for r in rows), [1,2])
        self.assertEqual(next(r for r in rows if r["support"] == 2)["actionable"], "0")
        self.config.min_support = 2
        self.assertEqual(len(reconcile([article], pool, self.config, Metric(), .8)), 1)

    def test_retries(self):
        self.config.retries = 1
        Server.malformed = True
        with patch("react_gpt.models.time.sleep"), self.assertRaises(RuntimeError):
            ModelClient(self.config).ask("family:a", "extract", "extract")
        self.assertEqual(len(Server.calls), 2)


class AlgorithmTests(unittest.TestCase):
    def test_ranking_prioritizes_quality_then_support_and_confidence(self):
        def row(name, sound, precise, support=1, confidence=1):
            return dict(actionable=name, SOUND=sound, PRECISE=precise,
                        support_fraction=support, avg_confidence=confidence)
        rows = [row("neither", "NO", "NO"), row("precise", "NO", "YES"),
                row("sound", "YES", "NO"), row("both-low", "YES", "YES", .5, .9),
                row("both-high", "YES", "YES", 1, .8), row("both-best", "YES", "YES", 1, .9)]
        ranked = rank_actionables(rows)
        self.assertEqual([r["actionable"] for r in ranked],
                         ["both-best", "both-high", "both-low", "sound", "precise", "neither"])
        self.assertEqual([r["rank"] for r in ranked], list(range(1, 7)))
        self.assertNotIn("rank", rows[0])
        self.assertEqual(rank_actionables([]), [])


    def test_complete_linkage_prevents_chaining(self):
        matrix = [[1, .9, .2], [.9, 1, .8], [.2, .8, 1]]
        groups = complete_linkage(matrix, .7)
        self.assertEqual(sorted(map(sorted, groups)), [[0, 1], [2]])
        self.assertEqual(complete_linkage([], .7), [])
        self.assertEqual(complete_linkage([[1]], .7), [[0]])

    def test_hybrid_weight_and_batched_pairs(self):
        class Vector:
            def __matmul__(self, other): return .5
        class Encoder:
            calls = 0
            def encode(self, texts, normalize_embeddings):
                self.calls += 1
                return [Vector() for _ in texts]
        metric = Similarity("lexical")
        metric.encoder = Encoder()
        self.assertAlmostEqual(metric.matrix(["mentor contributors", "mentor contributors"])[0][1], .7)
        metric.encoder.calls = 0
        values = metric.pairs([("mentor contributors", "mentor contributors"), ("mentor new contributors", "automate release testing")])
        self.assertAlmostEqual(values[0], .7)
        self.assertAlmostEqual(values[1], .3)
        self.assertEqual(metric.encoder.calls, 1)

    def test_strict_majority_even_panel(self):
        self.assertEqual(majority(["YES", "NO"]), "NO")
        self.assertEqual(majority(["YES", "YES", "NO"]), "YES")
        self.assertEqual(majority(["YES"]), "YES")

    def test_chunks_preserve_content(self):
        text = "abc def " * 100
        split = list(chunks(text, 31))
        self.assertEqual("".join(split), text)
        self.assertTrue(all(len(s) <= 31 for s in split))

    def test_response_validation(self):
        self.assertEqual(parse_response('```json\n{"recommendations": []}\n```', "extract"), [])
        self.assertEqual(parse_response('{"message":"NO ACTIONABLE CAN BE DERIVED"}', "extract"), [])
        for raw, kind in [('{"recommendations":null}', "extract"), ('{"verdict":"probably YES"}', "sound"),
                          ('{"categories":["invented"]}', "categories"), ('{}', "extract"),
                          ('{"features":["invented"]}', "features")]:
            with self.subTest(raw=raw), self.assertRaises(ValueError): parse_response(raw, kind)
        self.assertEqual(parse_response('{"features":["s_net_overlap"]}', "features"), ["s_net_overlap"])


if __name__ == "__main__":
    unittest.main()
