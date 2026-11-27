from pathlib import Path

import pytest

from cognates import dataset, distance, ipa, reverse, semantic, tree, wiktionary
from cognates.errors import PipelineStop


# Step 1 -------------------------------------------------------------------
def test_form_of_lemmatization():
    assert wiktionary.lemmatize("vaders", "nl").word == "vader"
    assert wiktionary.lemmatize("fathers", "en").word == "father"


def test_romanized_lookup_reaches_native_script():
    assert wiktionary.lemmatize("fadar", "got").word == "𐍆𐌰𐌳𐌰𐍂"


def test_missing_entry_stops():
    with pytest.raises(PipelineStop) as e:
        wiktionary.lemmatize("Väter", "nl")
    assert e.value.step == 1


def test_unsupported_language_stops():
    with pytest.raises(PipelineStop):
        wiktionary.lemmatize("Vater", "de")


def test_first_translation_and_english_identity():
    assert wiktionary.translation(wiktionary.lemmatize("vader", "nl")) == "father, male parent"
    assert wiktionary.translation(wiktionary.lemmatize("father", "en")) == "father"


# Step 2 -------------------------------------------------------------------
def test_ipa_prefers_phonemic_then_phonetic():
    assert wiktionary.ipa(wiktionary.lemmatize("vader", "nl"))[1] == "wiktionary-phonemic"
    assert wiktionary.ipa(wiktionary.lemmatize("fadar", "got"))[1] == "wiktionary-phonetic"


def test_ipa_clean_keeps_optional_sounds():
    assert ipa.clean("ˈfɑː.ðə(ɹ)") == "fɑːðəɹ"
    assert ipa.clean("pi.tŕ̩") == "pitr̩"


# Step 3 -------------------------------------------------------------------
def test_head_noun():
    assert semantic.head_noun("male parent") == "parent"
    assert semantic.head_noun("father (male parent)") == "father"
    assert semantic.head_noun("a father") == "father"


# Step 5 -------------------------------------------------------------------
def test_strip_vowels_keeps_glides_and_syllabic_consonants():
    assert ipa.strip_vowels("fɑːðəɹ") == "fðɹ"
    assert ipa.strip_vowels("pitr̩") == "ptr̩"
    assert ipa.strip_vowels("wajus") == "wjs"


def test_tree_chains():
    n = tree.load()
    chain = [n[s].target for s in reversed(tree.lineage(n, "British-English"))]
    assert chain[:4] == ["Common Germanic", "West Germanic", "Anglo-Frisian", "Old English"]
    assert n["High-German-Cosonant-Shift-and-Umlaut"].kind == "process"
    assert len(n["West-Germanic"].rules) == 5


def test_plan_direct_descent_and_pie():
    assert reverse.plan("sa", "sa").path1 == []
    p = reverse.plan("en", "nl")
    assert reverse.target_name(p.ancestor) == "West Germanic"
    assert reverse.target_name(reverse.plan("en", "sa").ancestor) == "Proto-Indo-European"


def test_per_position_tags():
    variants = reverse.tag_variants("tata", [{"sound": "a", "sources": ["a", "e"]}])
    assert sorted(variants) == sorted(["ta¹ta¹", "ta¹ta²", "ta²ta¹", "ta²ta²"])


def test_reverse_with_toy_stage(tmp_path, monkeypatch):
    (tmp_path / "Proto-Indo-Iranian.lsc").write_text(
        "Symbol tʃ\nmergers:\n    a² => e\n    a³ => o\npalatals:\n    tʃ => k / _ e\n")
    (tmp_path / "Proto-Indo-Iranian.mergers.yaml").write_text("- sound: a\n  sources: [a, e, o]\n")
    monkeypatch.setattr(reverse, "FAMILY_DIR", Path(tmp_path))
    forms, _ = reverse.reconstruct("tʃaːtʃa", ["Proto-Indo-Iranian"])
    assert forms == {"tʃaːke", "tʃaːtʃa", "tʃaːtʃo"}


# Step 6 -------------------------------------------------------------------
def test_symbol_mapping():
    assert ipa.to_ipa("*ph₂tḗr") == "pχteːr"
    assert ipa.to_ipa("*ǵénh₁os") == "ɟenʔos"
    assert ipa.to_ipa("*bʱréh₂tēr") == "bʰreχteːr"


def test_validation_halts_on_unknown_symbols():
    with pytest.raises(PipelineStop):
        ipa.validate("pXtr")


def test_distance_minimum_and_empty_roots():
    best = distance.closest({"fɑːðəɹ", "kat"}, {"fɑːðəɹ"})
    assert best.distance == 0 and best.root1 == "fðɹ"
    with pytest.raises(PipelineStop):
        distance.closest({"aː"}, {"ia"})


# Step 7 -------------------------------------------------------------------
def test_dataset_filters():
    anc, bor, com = dataset.classify({"etymology_templates": [
        {"name": "inh", "args": {"1": "en", "2": "gem-pro", "3": "*tanþs"}}]})
    assert anc == {("gem-pro", "*tanþs")} and not bor and not com
    _, bor, _ = dataset.classify({"etymology_templates": [{"name": "bor", "args": {"1": "en", "2": "la"}}]})
    assert bor
    _, _, com = dataset.classify({"etymology_templates": [
        {"name": "ety", "args": {"1": "en", "2": ":af", "3": "x", "4": "-y"}}]})
    assert com
    anc, _, _ = dataset.classify({"etymology_templates": [
        {"name": "ety", "args": {"1": "nl", "2": ":inh", "3": "dum:vāder<id:father>"}}]})
    assert anc == {("dum", "vāder")}


# Reversed rules: Common Germanic ------------------------------------------
def test_common_germanic_grimm_s_exception():
    forms, _ = reverse.reconstruct("ɣastiz", ["Common-Germanic"])
    assert "ɡʱostis" in forms and not any("sd" in f for f in forms)
    forms, _ = reverse.reconstruct("skipaz", ["Common-Germanic"])
    assert all(f.startswith(("sk", "sḱ")) for f in forms)
    forms, _ = reverse.reconstruct("faðeːr", ["Common-Germanic"])
    assert "pateːr" in forms


def test_common_germanic_stop_cluster_exception():
    forms, _ = reverse.reconstruct("naxts", ["Common-Germanic"])
    assert "nokts" in forms and not any("kd" in f for f in forms)
    forms, _ = reverse.reconstruct("xaftaz", ["Common-Germanic"])
    assert all("pt" in f for f in forms)


def test_common_germanic_palatovelars_and_labiovelars():
    assert "peḱu" in reverse.reconstruct("fexu", ["Common-Germanic"])[0]
    assert "ḱukis" in reverse.reconstruct("xuɣiz", ["Common-Germanic"])[0]  # palatal + Verner
    assert "kʷod" in reverse.reconstruct("xʷat", ["Common-Germanic"])[0]
    assert "ɡʷʱormos" in reverse.reconstruct("ɣʷarmaz", ["Common-Germanic"])[0]
    skw = reverse.reconstruct("skaðuz", ["Common-Germanic"])[0]
    assert {f[:2] for f in skw} == {"sk", "sḱ"}  # s-exception holds for the palatal branch too


def test_palatal_symbols_reach_panphon():
    assert distance.prepare("ǵʱeus") == "ɟʰs"
    assert distance.prepare("ḱuḱis") == "ccs"


def test_common_germanic_laryngeals_and_syllabic_resonants():
    cg = ["Common-Germanic"]
    assert "seh₁tis" in reverse.reconstruct("sæːðiz", cg)[0]      # eh₁ > ē
    assert "bʱreh₂teːr" in reverse.reconstruct("βroːθeːr", cg)[0]  # eh₂ > ō
    assert "h₁es" in reverse.reconstruct("iz", cg)[0]              # initial h₁e > i
    assert "pHteːr" in reverse.reconstruct("faðeːr", cg)[0]        # vocalized H > a
    assert "deḱn̩" in reverse.reconstruct("texun", cg)[0]           # n̥ > un
    # vocalization needs consonants on both sides; coloring needs word-initial position
    assert not any(f.startswith("H") for f in reverse.reconstruct("aɣaz", cg)[0])


# Reversed rules: West Germanic --------------------------------------------
def test_west_germanic_stage():
    wg = ["West-Germanic"]
    assert reverse.reconstruct("βero", wg)[0] >= {"βiru", "βizu", "βiruz"}  # a-mutation, rhotacism, final z
    assert reverse.reconstruct(ipa.clean("dag"), wg)[0] == {"ðaɣ", "ðaɣz"}     # ASCII g normalized
    assert reverse.reconstruct("kunnja", wg)[0] == {"kunja", "kunjaz"}        # degemination before j


def test_optional_ending_variants():
    v = reverse.tag_variants("dag", [{"optional_ending": "z"}])
    assert v == ["dag", "dagz"]


def test_west_germanic_long_e_to_long_a():
    assert "jæːr" in reverse.reconstruct("jaːr", ["West-Germanic"])[0]
    assert "jeh₁r" in reverse.reconstruct("jaːr", ["West-Germanic", "Common-Germanic"])[0]


# Reversed rules: Anglo-Frisian --------------------------------------------
def test_anglo_frisian_stage():
    af = ["Anglo-Frisian"]
    assert "ɡans" in reverse.reconstruct("ɡãːs", af)[0]      # nasal spirant law
    assert "fimf" in reverse.reconstruct("fĩːf", af)[0]      # ... with the m branch
    assert "xaim" in reverse.reconstruct("haːm", af)[0]      # ai > aː, plus h > x notation
    assert "kiːn" in reverse.reconstruct("tʃiːn", af)[0]     # palatalization
    assert reverse.reconstruct("dæɡ", af)[0] == {"daɡ"}      # brightening; no vowel endings (team decision)


def test_germanic_chain_reaches_pie():
    path = ["Anglo-Frisian", "West-Germanic", "Common-Germanic"]
    best = distance.closest(reverse.reconstruct("ɡãːs", path)[0], {"*ǵʰh₂ens"})
    assert best.distance == 0


# Reversed rules: Old English ----------------------------------------------
def test_old_english_stage():
    oe = ["Old-English"]
    assert reverse.reconstruct("ɡoːs", oe)[0] == {"ɡoːs", "ɡãːs", "ɡõːs"}  # denasalization, before fricative
    assert reverse.reconstruct("ɡoːd", oe)[0] == {"ɡoːd"}                    # no branch before a stop
    assert reverse.reconstruct("tʃild", oe)[0] == {"kild"}                   # palatalization
    assert reverse.reconstruct("bɑnd", oe)[0] == {"band"}                    # ɑ notation bridge


def test_old_english_chain_reaches_pie():
    path = ["Old-English", "Anglo-Frisian", "West-Germanic", "Common-Germanic"]
    assert distance.closest(reverse.reconstruct("muːθ", path)[0], {"*mn̥tós"}).distance == 0


def test_old_english_i_umlaut_and_breaking():
    oe = ["Old-English"]
    assert reverse.reconstruct("myːs", oe)[0] == {"muːs"}              # y > u
    assert "foːt" in reverse.reconstruct("feːt", oe)[0]                # eː > oː
    assert reverse.reconstruct("heɑrd", oe)[0] == {"hærd"}             # breaking eɑ > æ before r
    assert reverse.reconstruct("hæɑrd", oe)[0] == {"hærd"}             # Wiktionary spelling
    assert "erθa" in reverse.reconstruct("eorθe", oe)[0]               # eo > e before r
    assert reverse.reconstruct("fyllɑn", oe)[0] == {"fullan"}
    path = ["Old-English", "Anglo-Frisian", "West-Germanic", "Common-Germanic"]
    assert distance.closest(reverse.reconstruct("myːs", path)[0], {"*muHs"}).distance == 0


# Reversed rules: Midlands Middle English ----------------------------------
def test_midlands_middle_english_stage():
    me = ["Midlands-Middle-English"]
    assert "dohtor" in reverse.reconstruct("dowtər", me)[0]     # ow < oh before t; ə < o
    assert "ʃeːɑp" in reverse.reconstruct("ʃɛːp", me)[0]        # ɛː < eːɑ
    assert "nɑmɑ" in reverse.reconstruct("nɑːmə", me)[0]        # lengthening; ə < ɑ
    assert "fyːr" in reverse.reconstruct("fiːr", me)[0]         # iː < yː
    assert "hlɑːɣ" in reverse.reconstruct("lɔːw", me)[0]        # ɔːw < ɑːɣ, initial hl


def test_context_limited_tags():
    m = [{"sound": "e", "before": ["w"], "sources": ["e", "x"]}, {"sound": "ɡ", "initial": True, "sources": ["ɡ", "ɣ"]}]
    assert len(reverse.tag_variants("ɡeɡe", m)) == 2    # only the initial ɡ branches; no e precedes w
    assert len(reverse.tag_variants("ɡewe", m)) == 4


# Reversed rules: Early Modern English -------------------------------------
def test_early_modern_english_stage_and_great_vowel_shift():
    em = ["Early-Modern-English"]
    assert reverse.reconstruct("baɪt", em)[0] == {"biːt", "biht"}   # aɪ < iː, or ih before t (lost gh)
    assert reverse.reconstruct("haʊs", em)[0] == {"huːs"}           # aʊ < uː
    assert reverse.reconstruct("ɡuːs", em)[0] == {"ɡoːs"}           # uː < oː
    assert "nɑːm" in reverse.reconstruct("neɪm", em)[0]             # eɪ < ɑː
    assert reverse.reconstruct("miːt", em)[0] == {"meːt", "mɛːt"}   # meet/meat merger
    assert "foːt" in reverse.reconstruct("fʊt", em)[0]              # shortened oː
    assert reverse.reconstruct("bɜːɹd", em)[0] == {"berd", "bird", "burd"}
    assert "sjuɡər" in reverse.reconstruct("ʃʊɡəɹ", em)[0]          # yod coalescence


def test_english_chain_from_early_modern_reaches_pie():
    path = ["Early-Modern-English", "Midlands-Middle-English", "Old-English",
            "Anglo-Frisian", "West-Germanic", "Common-Germanic"]
    for word, root in [("maʊs", "*muHs"), ("fʊt", "*pṓds"), ("fɑːðəɹ", "*ph₂tḗr")]:
        assert distance.closest(reverse.reconstruct(word, path)[0], {root}).distance == 0


# Reversed rules: British English and the full English chain ---------------
def test_british_english_stage():
    br = ["British-English"]
    assert reverse.reconstruct("bɑːθ", br)[0] == {"bæθ", "bɑːθ", "bɑːɹθ"}   # BATH, lost r
    assert reverse.reconstruct("bɜːd", br)[0] == {"bɜːɹd"}                  # ɜː always had r
    assert reverse.reconstruct("dɪə", br)[0] == {"diːɹ"}
    assert reverse.reconstruct("fɑːðəɹ", br)[0] == {"fɑːðəɹ", "fɑːɹðəɹ"}    # no second linking r
    assert "ʍeɪl" in reverse.reconstruct("weɪl", br)[0]                      # ʍ > w
    assert reverse.reconstruct("ɡəʊt", br)[0] == {"ɡoʊt"}


def test_ipa_clean_strips_nonsyllabic_mark():
    assert ipa.clean("haʊ̯s") == "haʊs"


def test_full_english_chain_from_wiktionary():
    from cognates import pipeline
    path = reverse.path_to("en", None)
    assert len(path) == 7
    for word, root in [("father", "*ph₂tḗr"), ("mouse", "*muHs"), ("foot", "*pṓds"), ("whale", "*kʷalos")]:
        w = pipeline.prepare_word(word, "en")
        assert distance.closest(reverse.reconstruct(w.ipa, path)[0], {root}).distance == 0, word


def test_english_gap_fixes():
    assert "mʌðəɹ" in reverse.reconstruct("mʌðə", ["British-English"])[0]      # lettER
    assert "niht" in reverse.reconstruct("naɪt", ["Early-Modern-English"])[0]  # lost gh
    assert "dowtər" in reverse.reconstruct("dɔːtəɹ", ["Early-Modern-English"])[0]
    assert wiktionary.ipa(wiktionary.lemmatize("heart", "en"))[0] == "hɑːt"     # RP preferred


# Reversed rules: Sanskrit chain -------------------------------------------
def test_sanskrit_chain_stages():
    assert reverse.path_to("sa", None) == ["Vedic-Sanskrit", "Proto-Indo-Aryan", "Proto-Indo-Iranian"]
    assert "ḱlewos" in reverse.reconstruct("ɕɾɐʋɐs", reverse.path_to("sa", None))[0]   # śrávas
    assert "ǵaːnu" in reverse.reconstruct("d͡ʑɑːnu", reverse.path_to("sa", None))[0]   # jā́nu
    assert "dai" in "".join(reverse.reconstruct("deːʋɐ", ["Vedic-Sanskrit", "Proto-Indo-Aryan"])[0])  # e < ai (team)


def test_english_sanskrit_cognates_beat_controls():
    from cognates import pipeline
    cognate = pipeline.analyze("father", "en", "पितृ", "sa")[0]["phonetic_distance"]
    control = pipeline.analyze("father", "en", "अग्नि", "sa")[0]["phonetic_distance"]
    assert cognate == 0 and control > 5


# Reversed rules: Gothic ---------------------------------------------------
def test_gothic_stage():
    g = ["Gothic"]
    assert "ðaɣz" in reverse.reconstruct("daxs", g)[0]        # final devoicing, final s < z, d < ð
    assert "xlaiβz" in reverse.reconstruct("hlɛːfs", g)[0]    # ai digraph
    assert "ðuxtar" in reverse.reconstruct("dɔhtar", g)[0]    # breaking u > ɔ before h
    assert reverse.reconstruct("ɛz", g)[0] == {"aiz"}


def test_english_gothic_cognates_beat_controls():
    from cognates import pipeline
    assert pipeline.analyze("day", "en", "dags", "got")[0]["phonetic_distance"] == 0
    assert pipeline.analyze("wolf", "en", "juk", "got")[0]["phonetic_distance"] > 5


# Step 6 metric: jt_weighted_feature_edit_distance (team decision 2026-11-27)
def test_distance_uses_jt_metric_with_cheap_insertions():
    # one extra final consonant costs a flat 1.8125, whatever the consonant
    for extra in ("s", "n", "k"):
        assert distance.closest({"wulf" + extra}, {"wulf"}).distance == pytest.approx(1.8125)
    assert distance.closest({"taks"}, {"t"}).distance == pytest.approx(2 * 1.8125)
    # substitutions keep their feature weights
    assert distance.closest({"s"}, {"z"}).distance == pytest.approx(0.25)
    assert distance.closest({"k"}, {"t"}).distance == pytest.approx(2.125)


def test_forgive_switch_is_retired(tmp_path):
    from cognates import cli
    with pytest.raises(SystemExit):
        cli.main(["test", "father", "en", "fadar", "got", "--forgive-missing-endings"])


def test_table_output_path_and_skipped_log_location(tmp_path):
    from cognates import table
    assert table.skipped_path(table.TABLE_CSV) == table.SKIPPED_CSV
    assert table.skipped_path(tmp_path / "other.csv") == tmp_path / "other_skipped.csv"


def test_cli_accepts_output_option(tmp_path, monkeypatch):
    from cognates import cli, table
    seen = {}
    monkeypatch.setattr(table, "build", lambda *a, **k: (seen.update(k) or (k["output"], 0, 0)))
    cli.main(["table", "father", "en", "fadar", "got", "--output", str(tmp_path / "x.csv")])
    assert seen["output"] == tmp_path / "x.csv"


# LaTeX tree ---------------------------------------------------------------
def test_reconstruct_records_parents_and_chain(tmp_path, monkeypatch):
    (tmp_path / "Proto-Indo-Iranian.lsc").write_text(
        "Symbol tʃ\nmergers:\n    a² => e\n    a³ => o\npalatals:\n    tʃ => k / _ e\n")
    (tmp_path / "Proto-Indo-Iranian.mergers.yaml").write_text("- sound: a\n  sources: [a, e, o]\n")
    monkeypatch.setattr(reverse, "FAMILY_DIR", Path(tmp_path))
    forms, trace = reverse.reconstruct("tʃaːtʃa", ["Proto-Indo-Iranian"])
    assert trace[0].parents["tʃaːke"] == ["tʃaːtʃa"]
    assert reverse.chain("tʃaːke", trace, "tʃaːtʃa") == [("Proto-Indo-Iranian", "tʃaːke"), (None, "tʃaːtʃa")]


def test_tree_tex_for_father_pitr():
    from cognates import latex_tree, pipeline
    row, details = pipeline.analyze("father", "en", "पितृ", "sa")
    tex = latex_tree.tree_tex(row, details)
    for text in ["*pχtr̩", "Proto-Indo-European", "Common Germanic", "British English",
                 "Vedic Sanskrit", r"{\devafont पितृ}", r"\begin{forest}"]:
        assert text in tex, text


def test_tex_text_wraps_scripts_and_escapes():
    from cognates import latex_tree
    assert latex_tree.tex_text("𐍆𐌰𐌳𐌰𐍂 (fadar)") == r"{\gothfont 𐍆𐌰𐌳𐌰𐍂} (fadar)"
    assert latex_tree.tex_text("a_b~c") == r"a\_b\textasciitilde{}c"


def test_cli_tree_respects_threshold(tmp_path, capsys):
    from cognates import cli
    out = tmp_path / "t.tex"
    cli.main(["test", "father", "en", "अग्नि", "sa", "--tree", str(out)])
    assert not out.exists() and "no tree" in capsys.readouterr().out
    cli.main(["test", "father", "en", "अग्नि", "sa", "--tree", str(out), "--tree-threshold", "10"])
    assert out.exists()


def test_read_word_list_takes_language_from_extension(tmp_path):
    from cognates import table

    f = tmp_path / "list1.en"
    f.write_text("father\n\n  wolf \n", encoding="utf-8")
    assert table.read_word_list(f) == ("en", ["father", "wolf"])
    with pytest.raises(PipelineStop):
        table.read_word_list(tmp_path / "list1")
    bad = tmp_path / "list1.xx"
    bad.write_text("father\n")
    with pytest.raises(PipelineStop):
        table.read_word_list(bad)


def test_product_scores_every_ordered_pair_and_logs_stops(tmp_path):
    import csv
    from cognates import cli

    (tmp_path / "a.en").write_text("father\nzzzqx\n", encoding="utf-8")
    (tmp_path / "b.got").write_text("fadar\ndags\n", encoding="utf-8")
    out = tmp_path / "p.csv"
    cli.main(["product", str(tmp_path / "a.en"), str(tmp_path / "b.got"), "--output", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [(r["noun1"], r["translation2"]) for r in rows] == [("father", "father"), ("father", "day")]
    assert float(rows[0]["phonetic_distance"]) == 0.0
    skipped = list(csv.DictReader(open(tmp_path / "p_skipped.csv", encoding="utf-8")))
    assert [(s["word1"], s["word2"], s["step"]) for s in skipped] == [("zzzqx", "fadar", "1"), ("zzzqx", "dags", "1")]
