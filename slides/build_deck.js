// Builds the experiment deck: node slides/build_deck.js  (NODE_PATH must include pptxgenjs; figures from slides/make_assets.sh)
// One message per slide, Arial throughout, no title or divider slides. Figures come from assets_arial/
// (redrawn in Arial without figure titles).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/sv4956/.claude/remote/plugins/a4b17627/c1a46d3ed5b858d0/skills/pptx/scripts/apply_theme.js");

const ASSETS = path.join(__dirname, "assets_arial");
const SIZES = JSON.parse(fs.readFileSync(path.join(ASSETS, "sizes.json")));
const OUT = path.join(__dirname, "atom_matching_2026-10-26.pptx");

const THEME = {
  name: "Plain",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "333333", lt2: "F2F2F2",
    accent1: "1F5FA8", accent2: "B8642A", accent3: "7F7F7F", accent4: "A6ADB4", accent5: "1F8A83", accent6: "5B6B82",
    hlink: "1F5FA8", folHlink: "5B6B82",
  },
};
const HEX = THEME.colors;
const INK = "1A1A1A", GREY = "595959", RULE = "D9D9D9";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Comparing crystal structures across datasets with UMA and MACE atom embeddings";
pres.author = "sv4956";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.95, fontSize: 26, bold: true, color: INK, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: GREY, align: "right" },
});

// ---------- helpers ----------
let sec = null;
function section(title) { sec = title; pres.addSection({ title }); }
function slide(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
function img(s, file, x, y, w, h, name) {
  const [iw, ih] = SIZES[file];
  const k = Math.min(w / iw, h / ih), dw = iw * k, dh = ih * k;
  s.addImage({ path: path.join(ASSETS, file), x: x + (w - dw) / 2, y: y + (h - dh) / 2, w: dw, h: dh, objectName: name, altText: name });
}
// items: "plain text" or ["bold lead", "rest of the sentence"]
function bullets(s, items, x, y, w, h, opts = {}) {
  const runs = items.map((t, i) => {
    const base = { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: opts.space || 10 };
    if (Array.isArray(t)) return [{ text: t[0] + " ", options: { ...base, bold: true, breakLine: false } }, { text: t[1], options: { ...base, bullet: false } }];
    return [{ text: t, options: base }];
  }).flat();
  s.addText(runs, { x, y, w, h, fontSize: opts.size || 16, color: INK, valign: "top", margin: 0, isTextBox: true, objectName: opts.name || "points" });
}
function caption(s, text, x, y, w, h = 0.4) {
  s.addText(text, { x, y, w, h, fontSize: 12, color: GREY, valign: "top", margin: 0, isTextBox: true, objectName: "caption" });
}
function stats(s, items, x, y, w, opts = {}) {
  const cw = w / items.length;
  items.forEach(([v, l], i) => {
    s.addText(v, { x: x + i * cw, y, w: cw - 0.3, h: 0.6, fontSize: opts.size || 28, bold: true, color: C.accent1, valign: "bottom", margin: 0, isTextBox: true, objectName: "figure " + v });
    s.addText(l, { x: x + i * cw, y: y + 0.65, w: cw - 0.3, h: opts.lh || 0.7, fontSize: 13, color: GREY, valign: "top", margin: 0, isTextBox: true, objectName: "figure label " + v });
  });
}
function stack(s, items, x, y, w, gap = 1.45) {
  items.forEach(([v, l], i) => {
    s.addText(v, { x, y: y + i * gap, w, h: 0.55, fontSize: 26, bold: true, color: C.accent1, valign: "bottom", margin: 0, isTextBox: true, objectName: "figure " + v });
    s.addText(l, { x, y: y + i * gap + 0.6, w, h: 0.7, fontSize: 14, color: GREY, valign: "top", margin: 0, isTextBox: true, objectName: "figure label " + v });
  });
}
function table(s, header, rows, x, y, w, colW, opts = {}) {
  const fs = opts.size || 13;
  const head = header.map(t => ({ text: t, options: { bold: true, color: INK, fill: { color: HEX.lt2 }, fontSize: fs } }));
  const body = rows.map(r => r.map((t, j) => ({ text: String(t), options: { color: INK, fontSize: fs, bold: !!(opts.boldFirst && j === 0) } })));
  s.addTable([head, ...body], { x, y, w, colW, border: { type: "solid", pt: 0.5, color: RULE }, margin: 0.06, valign: "middle", objectName: opts.name || "table" });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt" };
function chart(title, extra = {}) {
  return {
    ...chartText, showTitle: true, title, titleFontSize: 13, titleColor: INK,
    catAxisLabelColor: INK, valAxisLabelColor: GREY, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: "E6E6E6", size: 0.75 }, catGridLine: { style: "none" },
    showValue: true, dataLabelFontSize: 11, dataLabelColor: INK, showLegend: false, ...extra,
  };
}

(async () => {
  let s;

  // ===== Summary =====
  section("Summary");
  s = slide("Summary", "Principal results. Atom embeddings: MACE features are dominated by element identity, UMA features by local environment. Matching: a geometric term (fused Gromov–Wasserstein with a periodic neighbour kernel) is required to preserve local connectivity; the choice of embedding is secondary. Structure distance: a size-intensive FGW formulation is invariant to supercell expansion. Dataset comparison: MOF-off is chemically disjoint from OMat24, and its closest OMat24 analogues are near-equilibrium AIMD frames.");
  [
    ["Atom embeddings.", "MACE atom embeddings are element-resolved (10-NN element agreement 92% for the MH1 readout, ≈100% for every interaction layer); UMA embeddings are organised by local environment (η²(element) = 0.25)."],
    ["Atom matching.", "Between a cell and its supercell, fused Gromov–Wasserstein (FGW) with a periodic neighbour kernel retains 69% of neighbour pairs, against 52% for embedding-only optimal transport (OT) and a ceiling of 74%. The embedding used has no significant effect."],
    ["Structure distance.", "A size-intensive FGW distance retrieves the same material across 2× and 4× supercells in 97% of cases."],
    ["MOF-off vs OMat24.", "The datasets share elements but not chemistry: 2 of 2,552 MOF-off compositions occur in OMat24, and 83–93% of MOF-off atoms lie in embedding regions that OMat24 samples at least ten times more sparsely."],
    ["Analogue search.", "Among 10M OMat24 structures, the closest analogue of every MOF query is a near-equilibrium AIMD frame; rattled 300/500/1000 frames distort C–H bonds (62–64% deviate by more than 0.15 Å)."],
  ].forEach(([lead, body], i) => {
    const y = 1.45 + i * 1.08;
    s.addText(String(i + 1), { x: 0.6, y, w: 0.5, h: 0.5, fontSize: 22, bold: true, color: C.accent1, margin: 0, isTextBox: true, valign: "top", objectName: "summary number " + (i + 1) });
    s.addText([{ text: lead + " ", options: { bold: true } }, { text: body }], { x: 1.2, y, w: 11.5, h: 0.95, fontSize: 16, color: INK, valign: "top", margin: 0, isTextBox: true, objectName: "summary point " + (i + 1) });
  });

  // ===== Data =====
  section("Data");
  s = slide("Four embedding representations cover 101M OMat24 and 13.5M external structures", "The shared archive (1.9 TB, 49,586 files) contains four representations. Per-structure vectors are Z-weighted sums of atom vectors and are therefore extensive. Per-atom banks exist only for UMA-128 and MH1-96. Coordinates are not part of the archive and are read from the source LMDB and trajectory files; bank row i corresponds to source frame i, which we verified. 468 files are group-restricted; the UMA banks affected were recovered from a combined bank. Frames with interatomic distances below 0.8 Å produce extreme embeddings and were excluded.");
  table(s, ["Representation", "Model", "Layer", "Width", "Per atom"], [
    ["UMA-128", "uma-s-1p2", "Input to the energy block", "128", "Yes"],
    ["OMAT0-128", "mace-omat-0-medium", "Last-layer invariants", "128", "No"],
    ["MH1-96", "mace-mh-1", "Readout hidden units, 6 heads", "96", "Yes"],
    ["MH1-1024", "mace-mh-1", "Invariants of both interaction layers", "1,024", "No"],
  ], 0.6, 1.5, 7.3, [1.45, 1.85, 2.55, 0.7, 0.75], { size: 13, name: "representations table" });
  table(s, ["Dataset", "Structures", "Per-atom sample"], [
    ["OMat24", "100.8M", "1M and 10M"], ["AM", "12.0M", "10%"], ["MP-ALOE", "910k", "10%"], ["MatPES", "388k", "10%"], ["MAD", "96k", "10%"], ["MOF-off (r²SCAN)", "80.6k", "10%"],
  ], 8.3, 1.5, 4.4, [1.75, 1.2, 1.45], { size: 13, name: "datasets table" });
  bullets(s, [
    ["Extensive structure vectors:", "H = Σᵢ Zᵢhᵢ grows with the number of atoms in the cell."],
    ["Coordinates:", "read from the source LMDB and trajectory files; bank row i corresponds to frame i."],
    ["Exclusions:", "468 group-restricted files (40.6 GB); 12,275 sr_10m frames with interatomic distances below 0.8 Å."],
  ], 0.6, 4.3, 12.1, 1.9, { size: 15 });

  // ===== Atom embeddings =====
  section("Atom embeddings");
  s = slide("MACE-MH1 atom embeddings are element-resolved; UMA embeddings encode local environment", "93,633 atoms from 5,000 random OMat24 sr_1m structures, 85 elements. The metric is the fraction of atoms whose ten nearest neighbours (Euclidean) vote for the correct element. Element identity explains 79% of the variance of the raw MH1 readout features and about 26% for UMA. A small number of high-variance UMA channels track the OMat24 subset label; per-channel standardisation removes this. Two-dimensional projections discard much of the element information for UMA.");
  s.addChart(pres.charts.BAR, [{ name: "Agreement", labels: ["Majority class", "UMA, PCA 2D", "UMA, UMAP 2D", "UMA, raw 128D", "UMA, PCA 25D", "UMA, standardised", "MH1, raw 96D"], values: [4, 12, 29, 63, 81, 83, 92] }],
    { x: 0.6, y: 1.4, w: 7.4, h: 5.3, barDir: "bar", chartColors: [HEX.accent1], ...chart("10-nearest-neighbour element agreement (%)", { valAxisMaxVal: 100, valAxisLabelFormatCode: "0", dataLabelPosition: "outEnd" }), objectName: "element agreement chart" });
  bullets(s, [
    ["η²(element):", "0.79 for MH1, 0.26 for UMA."],
    ["UMA scaling:", "a few high-variance channels track the OMat24 subset; standardisation raises agreement from 63% to 83%."],
    ["No clustering by structure:", "only 0.3% of neighbours are other-element atoms of the same structure."],
  ], 8.4, 1.6, 4.3, 4.6, { size: 15 });
  caption(s, "93,633 atoms, 5,000 OMat24 sr_1m structures.", 8.4, 6.3, 4.3);

  s = slide("In two dimensions, MH1 resolves element islands whereas UMA forms one continuous manifold", "UMAP of all 18.7M OMat24 sr_1m atoms with identical cuML settings for both models. First column: atom density. Remaining columns: one element family over the full map (grey). The UMA map is a single connected manifold in which families occupy overlapping bands ordered by coordination environment. The MH1 map consists of separate islands, each owned by one or a few elements. Refitting the UMA map on OMat24 atoms alone, rather than Aryan's joint fit with external data, does not change this (29% vs 30% element agreement).");
  img(s, "umap_families.png", 0.6, 1.5, 12.1, 3.6, "UMAP by element family");
  stats(s, [["86%", "2D element agreement, MH1"], ["29–30%", "2D element agreement, UMA (joint or OMat24-only fit)"], ["18.7M", "atoms per map; cuML UMAP, k = 15, min_dist = 0.1"]], 0.6, 5.45, 12.1);

  const VARS = ["UMA-128", "MH1-96 readout", "MH1-16 omat_pbe head", "MH1-512 layer 1", "MH1-512 layer 2", "MH1-1024 both", "OMAT0-128 layer 2", "OMAT0-256 both"];
  s = slide("Every MACE interaction layer is effectively an element label", "Per-atom invariants of both interaction layers of MACE-MH1 (omat_pbe head) and MACE-OMAT-0 were extracted for 50,188 OMat24 sr_1m structures (927,070 atoms). Their Z-weighted sums reproduce the archived per-structure vectors to a relative error of 3 × 10⁻⁸. All variants were standardised per channel. Interaction-layer invariants give near-perfect element agreement in both the full space and a 2D UMAP. The readout hidden units are the MACE features with the most environment information; the 16 units of the omat_pbe head behave like the full readout.");
  s.addChart(pres.charts.BAR, [
    { name: "Full space", labels: VARS, values: [82, 93, 89, 100, 100, 100, 99, 99] },
    { name: "2D UMAP", labels: VARS, values: [41, 81, 73, 99, 98, 98, 97, 97] },
  ], { x: 0.6, y: 1.4, w: 7.6, h: 5.4, barDir: "bar", barGapWidthPct: 60, chartColors: [HEX.accent4, HEX.accent1],
       ...chart("10-nearest-neighbour element agreement (%)", { valAxisMaxVal: 100, valAxisLabelFormatCode: "0", dataLabelPosition: "outEnd", dataLabelFontSize: 10, catAxisLabelFontSize: 11, showLegend: true, legendPos: "b", legendFontSize: 12, catAxisOrientation: "maxMin" }), objectName: "variant agreement chart" });
  table(s, ["Variant", "η²(element)"], [["UMA-128", "0.25"], ["MH1-96 readout", "0.41"], ["MH1-16 omat_pbe head", "0.40"], ["MH1-512 layer 1", "0.81"], ["MH1-512 layer 2", "0.68"], ["OMAT0-128 layer 2", "0.53"]],
    8.6, 1.55, 4.1, [2.7, 1.4], { size: 13, name: "eta table" });
  caption(s, "50,188 OMat24 sr_1m structures (927,070 atoms); features standardised per channel. The extracted descriptors reproduce the archived per-structure vectors to 3 × 10⁻⁸.", 8.6, 4.55, 4.1, 1.2);

  s = slide("Only UMA retains a connected representation in two dimensions", "UMAP of each variant on the same 925,350 atoms with the standard settings. Copper: transition metals; teal: non-metals; grey: all other atoms. UMA places each family in a broad connected region. The MH1 readout and omat_pbe head combine element islands with shared regions. Every interaction-layer variant fragments the atoms into compact single-element islands. The UMA agreement here (41%) exceeds that of the full map (29%) because the features were standardised.");
  img(s, "umap_variants_compact.png", 0.6, 1.35, 9.0, 5.6, "UMAP per variant");
  bullets(s, [
    ["UMA:", "families form broad, connected regions ordered by environment."],
    ["MH1 readout:", "element islands plus regions shared by similar environments."],
    ["Interaction layers:", "one compact island per element; environment variation is confined within islands."],
  ], 9.9, 1.6, 2.8, 5.0, { size: 14 });

  s = slide("MOF-off atoms occupy regions of embedding space that OMat24 samples sparsely", "Joint UMAP per model over 20.9M atoms: all OMat24 sr_1m atoms and the 10% atom samples of MOF-off, MAD, MP-ALOE and MatPES. AM has no MH1 atom bank and was excluded; 557 MAD atoms with non-finite MH1 features were removed. Each column highlights one dataset. The enrichment measure is the fraction of a dataset's atoms in map pixels where that dataset is at least ten times more abundant, relative to its size, than OMat24. MOF-off atoms lie predominantly in such regions; MP-ALOE and MatPES lie almost entirely within the OMat24 footprint. MAD is intermediate and includes non-periodic molecules and clusters.");
  img(s, "joint_by_dataset.png", 0.6, 1.4, 8.6, 4.3, "joint UMAP by dataset");
  table(s, ["Dataset", "UMA", "MH1"], [["MOF-off", "93%", "83%"], ["MAD", "42%", "43%"], ["MatPES", "11%", "14%"], ["MP-ALOE", "7%", "10%"]], 9.6, 1.6, 3.1, [1.5, 0.8, 0.8], { size: 14, name: "enrichment table" });
  caption(s, "Fraction of each dataset's atoms in map pixels where it is ≥ 10× enriched relative to OMat24.", 9.6, 3.6, 3.1, 0.9);
  caption(s, "Joint fit over 20.9M atoms: OMat24 sr_1m and 10% samples of MOF-off, MAD, MP-ALOE and MatPES. Rows: UMA, MACE-MH1.", 0.6, 5.95, 8.6, 0.6);

  s = slide("The six OMat24 generation protocols populate nearly the same regions", "Same joint maps, with OMat24 atoms separated by generation protocol. The footprints are nearly identical in both models; the protocols differ far less from each other than MOF-off differs from OMat24. Rattled 300, 500 and 1000 frames leave some regions unpopulated that AIMD and rattled-relax frames occupy, consistent with the bond distortion described in the analogue search.");
  img(s, "joint_by_omat_subset.png", 0.6, 1.5, 12.1, 4.2, "joint UMAP by OMat24 subset");
  caption(s, "Each column shows one protocol over all atoms (grey). Rows: UMA, MACE-MH1. Rattled 300/500/1000 frames leave regions unpopulated that AIMD and rattled-relax frames occupy.", 0.6, 5.95, 12.1, 0.7);

  s = slide("UMA embeddings vary smoothly with force magnitude and atomic volume", "Each pixel of the joint maps is coloured by the median percentile of an atomic or structural attribute; percentiles are computed over all 20.9M atoms so that the OMat24 majority does not compress the colour scale, and the colour bars give the corresponding values. UMA shows gradients across the entire manifold, most clearly in volume per atom. In MH1, the same gradients appear within element islands. Large cells, predominantly MOF-off, are distinct in both maps.");
  img(s, "joint_by_attribute.png", 0.6, 1.35, 8.4, 5.55, "joint UMAP by attribute");
  bullets(s, [
    ["Force:", "UMA shows a gradient across the main manifold (5th–95th percentile 0.13–6.0 eV/Å); in MH1 it varies within islands."],
    ["Cell size:", "the largest cells, mainly MOF-off (median 170 atoms), form distinct regions in both maps."],
    ["Volume per atom:", "the strongest UMA gradient, separating dense from open frameworks."],
    ["Median |F|:", "1.1 eV/Å in OMat24, 1.8 eV/Å in MOF-off, 0.5–0.6 eV/Å in MAD, MP-ALOE and MatPES."],
  ], 9.3, 1.5, 3.4, 5.4, { size: 14 });

  // ===== Cell-size invariance =====
  section("Cell-size invariance");
  s = slide("A benchmark of 94 materials, each as a primitive cell and a 2× or 4× rattled supercell", "OMat24 derives several structures from each Alexandria parent. The rattled-relax structure uses a primitive-sized cell; the rattled-300 structure uses a 2× or 4× supercell with random displacements. Pairs were retained when the volume per atom agrees within 5% and pymatgen's StructureMatcher maps the supercell onto the small cell, which provides the ground-truth atom correspondence. 76 of 160 held-out candidates failed alignment and were excluded. Ten curated pairs were used for development and 84 held out for evaluation.");
  stats(s, [["1,212", "materials with both a relaxed cell and a rattled-300 supercell"], ["10 + 84", "development and held-out pairs with a verified alignment"], ["0.6 Å", "median displacement of supercell atoms from ideal sites"]], 0.6, 1.5, 12.1, { lh: 0.8 });
  bullets(s, [
    ["Pairs:", "rattled-relax cell (2–8 atoms) and rattled-300 supercell from the same Alexandria parent, with volume per atom within 5%."],
    ["Ground truth:", "pymatgen StructureMatcher alignment of the supercell onto the small cell; 76 of 160 held-out candidates could not be aligned and were excluded."],
    ["Protocol:", "parameters were fixed on the 10 development pairs and evaluated once on the 84 held-out pairs."],
  ], 0.6, 3.4, 12.1, 3.2, { size: 16, space: 14 });

  s = slide("Extensive sums fail across cell sizes; normalisation or optimal transport restores invariance", "Retrieval test on 300 random pairs: each small cell must rank its own supercell first among 300 supercells. The stored Z-weighted sum scales with the number of atoms and fails. Division by the total atomic number is the simplest correction and performs best for UMA; optimal transport between the atom sets, with uniform weights 1/N, performs best for MH1. Across all 1,212 pairs the task is harder because many supercells are also strained.");
  s.addChart(pres.charts.BAR, [
    { name: "UMA", labels: ["Stored sum H", "H / ΣZ", "Mean of atoms", "Optimal transport"], values: [4, 64, 22, 51] },
    { name: "MACE-MH1", labels: ["Stored sum H", "H / ΣZ", "Mean of atoms", "Optimal transport"], values: [2, 61, 61, 81] }],
    { x: 0.6, y: 1.4, w: 7.6, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1], ...chart("Top-1 retrieval of the matching supercell (%, 300 pairs)", { valAxisMaxVal: 100, valAxisLabelFormatCode: "0", dataLabelPosition: "outEnd", showLegend: true, legendPos: "b", legendFontSize: 12 }), objectName: "retrieval chart" });
  bullets(s, [
    ["Stored sums:", "extensive in N; 2–4% top-1."],
    ["H / ΣZ:", "best for UMA (64%)."],
    ["Optimal transport:", "atom sets with weights 1/N; best for MH1 (81%)."],
    ["All 1,212 pairs:", "57% (UMA) and 50% (MH1) with H / ΣZ, owing to additional strain."],
  ], 8.6, 1.6, 4.1, 5.0, { size: 15 });

  // ===== Atom matching =====
  section("Atom matching");
  s = slide("Embedding-only OT assigns elements correctly but does not preserve local connectivity", "Optimal transport between the two sets of atom embeddings. Grey segments join a supercell atom to its true partner; red segments indicate an incorrect assignment, dashed segments a symmetry-equivalent copy. Element assignment is nearly perfect, but OT has no geometric term, and neighbour preservation is indistinguishable from random assignment within each element.");
  img(s, "ot_two.png", 0.6, 1.3, 5.4, 5.6, "OT mapping");
  stack(s, [["98–100%", "atoms assigned to the correct element"], ["62% / 53%", "exact site, UMA / MH1 (random within element: 49%)"], ["50–54%", "neighbour pairs preserved (random within element: 51%)"]], 6.6, 1.45, 6.1);
  bullets(s, [
    ["BaLiTb:", "every error is a symmetry-equivalent site."],
    ["KLiCl₂:", "inequivalent K and Cl sites are exchanged over 3–4 Å."],
  ], 6.6, 5.85, 6.1, 1.0, { size: 14, space: 6 });

  s = slide("Fused Gromov–Wasserstein adds a periodic neighbour-kernel term to the transport cost", "FGW combines the embedding cost of OT with a term that penalises assigning neighbouring atoms in one structure to non-neighbouring atoms in the other. The intra-structure similarity C is a Gaussian neighbour kernel summed over all periodic images. In the figure, grey bonds are preserved and red bonds are broken by the assignment; OT breaks many bonds, whereas FGW approaches the true correspondence. α = 0 recovers OT; α = 0.1–0.25 performed best on the development pairs.");
  s.addText([
    { text: "min", options: { breakLine: false } }, { text: "T", options: { subscript: true } },
    { text: "  (1 − α) Σ M", options: {} }, { text: "ij", options: { subscript: true } }, { text: " T", options: {} }, { text: "ij", options: { subscript: true } },
    { text: "  +  α Σ (C", options: {} }, { text: "1", options: { subscript: true } }, { text: "[i,k] − C", options: {} }, { text: "2", options: { subscript: true } }, { text: "[j,l])² T", options: {} }, { text: "ij", options: { subscript: true } }, { text: " T", options: {} }, { text: "kl", options: { subscript: true } },
  ], { x: 0.6, y: 1.35, w: 12.1, h: 0.55, fontSize: 20, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "fgw objective" });
  caption(s, "M: squared embedding distance between atoms i and j.   C: Gaussian neighbour kernel exp(−d²/2σ²) summed over all periodic images.   T: transport plan.", 0.6, 1.95, 12.1, 0.4);
  img(s, "bonds_two.png", 0.6, 2.45, 12.1, 4.35, "neighbour pairs preserved");

  s = slide("Three modifications were required to apply FGW to periodic crystals", "Applied directly, FGW failed. A primitive cell and its supercell have inconsistent kernels because each neighbour appears once in the small cell but is distributed over images in the supercell. Minimum-image distances capture only part of the neighbourhood in thin cells. The objective is non-convex, and two initialisations did not move the solution away from OT.");
  table(s, ["Modification", "Problem addressed", "Effect"], [
    ["Tile the small cell with a lattice-only integer supercell matrix", "Kernels of a cell and its supercell are not comparable", "True correspondence preferred by the objective in 10/10 pairs (previously 3/10)"],
    ["Sum the neighbour kernel over all periodic images", "Minimum-image distances retained 23–66% of each neighbourhood", "MH1 exact-site rate 45% → 55%"],
    ["100 random same-element initialisations", "Non-convex objective; two starts remained at the OT solution", "Neighbour preservation close to the true correspondence"],
  ], 0.6, 1.6, 12.1, [4.1, 4.0, 4.0], { size: 15, name: "modifications table", boldFirst: true });

  s = slide("On 84 held-out materials, FGW preserves 69% of neighbour pairs against 52% for OT", "Parameters were fixed on the ten development pairs, so the 84 held-out materials constitute an independent test. Differences were assessed per material with bootstrap confidence intervals and Holm-corrected Wilcoxon signed-rank tests. Two effects are significant: FGW preserves more neighbour pairs than OT, and embedding-only GW is worse than OT. Neither the choice of α nor the choice of UMA, MH1 or their concatenation has a significant effect.");
  s.addChart(pres.charts.BAR, [{ name: "Preserved", labels: ["GW, MH1", "GW, UMA+MH1", "GW, UMA", "Random within element", "OT, MH1", "OT, UMA+MH1", "OT, UMA", "FGW, UMA", "FGW, MH1", "FGW, UMA+MH1", "True correspondence"], values: [50, 51, 51, 51, 51, 51, 52, 68, 69, 69, 74] }],
    { x: 0.6, y: 1.4, w: 7.3, h: 5.4, barDir: "bar", chartColors: [HEX.accent1], ...chart("Neighbour pairs preserved (%, 84 held-out materials)", { valAxisMaxVal: 100, valAxisLabelFormatCode: "0", dataLabelPosition: "outEnd", catAxisLabelFontSize: 11 }), objectName: "held-out chart" });
  bullets(s, [
    ["Significant:", "FGW vs OT, +16–18 points (better for 75–79 of 84 materials)."],
    ["Significant:", "embedding-only GW is worse than OT by 12–15 points in exact-site rate."],
    ["Not significant:", "α = 0.1 vs 0.25; UMA vs MH1 vs UMA+MH1."],
    ["Supercell size:", "4× pairs are easier (81–86% exact site) than 2× pairs (57–66%)."],
  ], 8.3, 1.6, 4.4, 5.0, { size: 15 });
  caption(s, "Bootstrap CIs and Holm-corrected Wilcoxon signed-rank tests, per material.", 8.3, 6.35, 4.4);

  s = slide("Matching accuracy is insensitive to the choice of embedding layer", "All eight per-atom variants on the 94 benchmark pairs, with OT and with FGW (periodic kernel, α = 0.25, 30 initialisations), features standardised per channel. Element-pure layers remove the remaining element errors, but exact-site rates change little. FGW adds 15–18 points of neighbour preservation for every variant. Relative to the MH1 readout, no variant differs significantly in exact-site rate after Holm correction for seven comparisons; the largest difference, MH1 layer 2 (+3 points, p = 0.07), is within noise.");
  table(s, ["Variant", "Width", "Correct element, OT / FGW", "Exact site, OT / FGW", "Neighbours preserved, OT / FGW"], [
    ["UMA-128", "128", "96.9 / 97.6", "67.3 / 69.9", "52.3 / 67.2"],
    ["MH1-96 readout", "96", "98.7 / 98.9", "68.7 / 70.6", "52.1 / 68.3"],
    ["MH1-16 omat_pbe head", "16", "97.1 / 98.4", "66.7 / 67.9", "51.1 / 68.1"],
    ["MH1-512 layer 1", "512", "100 / 100", "69.0 / 70.6", "52.6 / 68.4"],
    ["MH1-512 layer 2", "512", "100 / 100", "70.6 / 73.8", "50.4 / 68.5"],
    ["MH1-1024 both", "1,024", "100 / 100", "69.9 / 71.5", "53.6 / 69.0"],
    ["OMAT0-128 layer 2", "128", "99.5 / 99.7", "70.6 / 70.5", "51.8 / 68.4"],
    ["OMAT0-256 both", "256", "99.7 / 100", "68.4 / 70.5", "51.6 / 68.0"],
  ], 0.6, 1.5, 12.1, [2.9, 1.0, 2.7, 2.6, 2.9], { size: 14, name: "variant matching table" });
  caption(s, "Percentages over 94 materials (10 development, 84 held-out). FGW: periodic kernel, α = 0.25, 30 initialisations. No difference from the MH1 readout is significant after Holm correction.", 0.6, 5.75, 12.1, 0.6);

  s = slide("Embedding-only Gromov–Wasserstein fails because no term anchors element identity", "Four further observations. Gromov–Wasserstein on intra-structure embedding distances alone has no term linking elements across the two structures and exchanges entire elements. Concatenating UMA and MH1, each block scaled by 1/√width, matches the better single model but is never significantly better. Symmetry is absent in the rattled data. The cost per pair is small enough for large-scale screening.");
  table(s, ["Observation", "Evidence"], [
    ["Embedding-only GW exchanges entire elements", "BaLiTb across models: 0% correct element; the GW objective prefers an incorrect assignment in 49 of 50 fits"],
    ["UMA + MH1 concatenation gives no measurable gain", "Equal to the better single model, never significantly better (blocks scaled by 1/√width)"],
    ["Symmetry is lost in the rattled data", "All rattled-300 supercells are P1 at 0.5 Å tolerance; 15–19% of small cells retain the parent space group"],
    ["The method is inexpensive", "Per pair on one core: OT 0.1 ms; one FGW solve 0.3 ms; 102 initialisations ≈ 30 ms plus 40–60 ms setup"],
  ], 0.6, 1.6, 12.1, [4.4, 7.7], { size: 15, name: "observations table", boldFirst: true });

  // ===== MOF-off vs OMat24 =====
  section("MOF-off vs OMat24");
  s = slide("MOF-off and OMat24 share elements but not chemistry", "Only two MOF-off compositions occur in a 10M-structure OMat24 sample, a praseodymium formate and an iron oxalate. All MOF-off elements occur in OMat24, but OMat24 carbon and hydrogen are almost exclusively in inorganic environments. In UMA space, about one third of MOF-off atoms are farther from any OMat24 atom than 99% of OMat24 atoms are from their nearest OMat24 neighbour. MH1, being element-dominated, shows a much smaller gap.");
  stack(s, [["2 of 2,552", "MOF-off compositions present in 10M OMat24 structures"], ["99.4% vs 0.11%", "structures containing both C and H (MOF-off vs OMat24)"], ["116 vs 12", "median atoms per structure"]], 0.6, 1.45, 4.3, 1.75);
  s.addChart(pres.charts.BAR, [
    { name: "UMA", labels: ["C", "N", "Cu", "H", "O"], values: [68, 28, 22, 5, 3] },
    { name: "MACE-MH1", labels: ["C", "N", "Cu", "H", "O"], values: [0.3, 17, 2, 0.1, 0] }],
    { x: 5.3, y: 1.4, w: 7.4, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1], ...chart("MOF-off atoms outside the OMat24 range, by element (%)", { valAxisMaxVal: 80, valAxisLabelFormatCode: "0", dataLabelFormatCode: "0", dataLabelPosition: "outEnd", showLegend: true, legendPos: "b", legendFontSize: 12 }), objectName: "outside range chart" });
  caption(s, "Outside: nearest OMat24 atom farther than the 99th percentile of OMat24 nearest-neighbour distances (20,000 atoms per set).", 5.3, 6.45, 7.4, 0.5);

  s = slide("A size-intensive FGW distance identifies the same material across cell sizes", "For arbitrary pairs, one structure cannot be tiled into the other, so the geometric term must be intensive. Scaling the periodic kernel by the number of atoms makes a supercell's kernel equal to the primitive cell's on average. Feature and kernel scales are fixed globally so that distances are comparable across a matrix. With σ = 3 Å, each small cell retrieves its own supercell among 94 in 97% of cases; a narrow 1 Å kernel is too sensitive to rattling.");
  bullets(s, [
    ["Intensive kernel:", "K = N · Σ over images exp(−d²/2σ²), normalised by a single global constant k₀."],
    ["Global scaling:", "feature (m₀) and kernel (k₀) scales are fixed once, so all distances in a matrix are comparable."],
    ["Validation:", "each of 94 small cells must retrieve its own 2× or 4× supercell."],
  ], 0.6, 1.6, 5.2, 4.8, { size: 16, space: 14 });
  s.addChart(pres.charts.BAR, [{ name: "Top-1", labels: ["σ = 1 Å", "σ = 2 Å", "Features only", "σ = 3 Å"], values: [39, 79, 95, 97] }],
    { x: 6.2, y: 1.4, w: 6.5, h: 5.4, barDir: "col", chartColors: [HEX.accent1], ...chart("Top-1 retrieval of the matching supercell (%, α = 0.5, 94 pairs)", { valAxisMaxVal: 100, valAxisLabelFormatCode: "0", dataLabelPosition: "outEnd" }), objectName: "intensive validation chart" });

  s = slide("MOF-off forms a distinct block in the pairwise structure-distance matrix", "Pairwise structure FGW distances (UMA+MH1 features, σ = 3 Å, α = 0.5) for 100 MOF-off and 1,000 random OMat24 structures, grouped by dataset and OMat24 protocol. MOF-off forms a separate block. Random OMat24 structures differ mainly by composition, so the protocols are hard to distinguish, except that AIMD 3000 K frames are slightly displaced and are the closest protocol to MOF-off.");
  img(s, "grouped_1100.png", 0.6, 1.35, 12.1, 4.95, "grouped distance matrix");
  caption(s, "Median distances: MOF–MOF 1.06; OMat24–OMat24 0.41; MOF–OMat24 1.60. 98 of 100 MOFs are closer to another MOF than to any OMat24 structure.", 0.6, 6.4, 12.1, 0.45);

  s = slide("The closest MOF–OMat24 pairs share metal centres, halides or organic fragments", "Three of the closest MOF-off to OMat24 pairs. The grids on the right show where the atoms of each MOF element are assigned by the FGW plan. Shared elements are matched to each other; elements absent from the OMat24 structure are assigned to the most similar available element.");
  img(s, "closest_pairs_3.png", 0.6, 1.3, 6.5, 5.6, "closest pairs");
  bullets(s, [
    ["Cu–I framework and H₁₂CuI₆:", "closest pair overall; I → I 100%, Cu → Cu 68%."],
    ["Organic crystal C₃H₅N₅O:", "the only C- and H-containing structure in the 1,000-structure sample; nearest neighbour of 80 of 100 MOFs."],
    ["Ba(OH)₂:", "N and O are assigned to O, S to Ba; absent elements map to the most similar available element."],
  ], 7.5, 1.6, 5.2, 5.0, { size: 15, space: 14 });

  // ===== 10M search =====
  section("Analogue search");
  s = slide("A two-stage search ranks 10M OMat24 structures for each of 10 MOF queries", "Exact FGW against 10M structures for 10 queries would require about 360 core-hours. A surrogate score, combining feature OT with a one-dimensional Wasserstein distance between per-atom neighbour-density distributions, ranks all structures first. It correlates with FGW at Spearman ρ = 0.90 and placed 99% of the FGW top-10 within its own top 10% on a validation set. The top 50,000 candidates per query were then re-ranked with exact FGW.");
  const flow = [["10 MOF queries", "10 distinct metals; 42–276 atoms"], ["Surrogate prefilter", "feature OT + 1D W₂ on neighbour densities; ρ = 0.90 with FGW"], ["Top 50,000 per query", "0.5% of the sample; 168,736 distinct structures"], ["Exact FGW re-ranking", "500,000 pairs; UMA features, σ = 3 Å, α = 0.5"]];
  flow.forEach(([head, body], i) => {
    const x = 0.6 + i * 3.1;
    s.addText([{ text: head, options: { bold: true, breakLine: true, fontSize: 16, color: INK } }, { text: body, options: { fontSize: 14, color: GREY } }],
      { x, y: 1.6, w: 2.6, h: 1.5, valign: "top", margin: 0, isTextBox: true, objectName: "stage " + (i + 1) });
    if (i < flow.length - 1) s.addText("→", { x: x + 2.6, y: 1.55, w: 0.5, h: 0.5, fontSize: 22, color: C.accent1, align: "center", margin: 0, isTextBox: true, objectName: "stage arrow " + (i + 1) });
  });
  s.addShape(pres.shapes.LINE, { x: 0.6, y: 3.45, w: 12.1, h: 0, line: { color: RULE, width: 0.75 }, objectName: "divider" });
  stats(s, [["46", "core-hours for the prefilter (400 Slurm tasks)"], ["12,275", "unphysical frames excluded (atoms closer than 0.8 Å)"], ["≤ 520", "prefilter rank of every FGW top-20 match, for 7 of 10 queries"]], 0.6, 3.8, 12.1, { size: 32, lh: 0.8 });
  caption(s, "For the three largest MOFs (202–276 atoms), some top-20 matches originated from prefilter ranks of 11,000–35,000; their best matches are poor in any case.", 0.6, 5.9, 12.1, 0.6);

  s = slide("Chemically consistent analogues are recovered for about half of the queries", "Three of the ten queries. The Co MOF is matched to a manganese coordination compound with the metal assigned to the metal. The manganese formate is matched to a lithium manganese oxalate. The Zn MOF illustrates the main failure mode: a 3000 K hydrocarbon frame ranks first because 80% of the MOF atoms are C or H and every atom carries equal weight, so the metal is assigned to hydrogen.");
  img(s, "best_matches_3.png", 0.6, 1.3, 6.5, 5.6, "best matches");
  bullets(s, [
    ["Co MOF and Mn complex:", "Co → Mn 100%; ligand C, N and O on C, N and O/S."],
    ["Mn formate and Mn oxalate:", "Mn → Mn 95%, O → O 100%."],
    ["Zn MOF and a 3000 K hydrocarbon:", "metal, N and O assigned to H or C; uniform atom weights allow C and H to outweigh the metal."],
    ["No counterpart:", "the Ag, Ni and Ce MOFs (202–276 atoms) match only large organosilicon or organometallic cells."],
  ], 7.5, 1.6, 5.2, 5.2, { size: 15, space: 12 });

  const subsets = ["AIMD 1000 K", "AIMD 3000 K", "rattled relax", "rattled 300", "rattled 500", "rattled 1000"];
  s = slide("Rattling distorts C–H bonds; near-equilibrium frames match MOFs best", "With composition controlled (only OMat24 structures containing C and H), AIMD and rattled-relax frames are equally close to the MOFs on embedding features alone, whereas rattled 300, 500 and 1000 frames are 25–30% farther. Rattling displaces atoms independently and stretches C–H bonds, which UMA encodes as different environments. The three rattling labels distort hydrogen to the same extent.");
  s.addChart(pres.charts.BAR, [{ name: "Distorted", labels: subsets, values: [6, 20, 7, 64, 62, 63] }],
    { x: 0.6, y: 1.4, w: 6.0, h: 4.9, barDir: "col", chartColors: [HEX.accent1], ...chart("C–H bonds deviating > 0.15 Å from 1.09 Å (%)", { valAxisMaxVal: 80, valAxisLabelFormatCode: "0", dataLabelPosition: "outEnd", catAxisLabelFontSize: 10 }), objectName: "C-H chart" });
  s.addChart(pres.charts.BAR, [{ name: "Distance", labels: subsets, values: [1.68, 1.65, 1.67, 2.15, 2.06, 2.20] }],
    { x: 6.7, y: 1.4, w: 6.0, h: 4.9, barDir: "col", chartColors: [HEX.accent4], ...chart("Median MOF–subset distance, features only (α = 0)", { valAxisMinVal: 0, valAxisMaxVal: 2.5, valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", catAxisLabelFontSize: 10 }), objectName: "subset distance chart" });
  caption(s, "Both panels: OMat24 structures containing C and H only. Distances: 60 structures per protocol.", 0.6, 6.4, 12.1, 0.4);

  s = slide("For every query, the closest match lies in an AIMD subset", "Each cell is the best match of a query within one OMat24 protocol, expressed relative to that query's best match overall. AIMD subsets provide the closest and chemically most reasonable matches. Rattled-relax matches are 9–43% farther and often chemically consistent, for example a copper thiocyanate-type compound for the Cu MOF. Rattled 300, 500 and 1000 subsets return small metallic cells without chemical relationship to the query, 1.2–1.9 times farther; restricting to C- and H-containing structures does not close the gap.");
  img(s, "best_per_subset.png", 0.6, 1.35, 12.1, 5.6, "best match per subset");

  s = slide("With composition controlled, rattled organic frames cluster together and away from MOF-off", "Balanced matrix of 60 structures per group, all containing C and H, with the same global scaling as the search. Rattled 300, 500 and 1000 organic frames are closer to one another than to any other protocol, irrespective of their nominal temperature. With geometry included, AIMD 3000 K is the protocol closest to MOF-off, largely because its large, dense cells resemble MOF cells.");
  img(s, "grouped_organic_a05.png", 0.6, 1.35, 12.1, 4.95, "organic grouped matrix");
  caption(s, "Features only (α = 0): MOF–AIMD 1000 K, 3000 K and rattled relax 1.65–1.68; MOF–rattled 300/500/1000 2.06–2.20; MOF–MOF 0.57.", 0.6, 6.4, 12.1, 0.45);

  // ===== Conclusions =====
  section("Conclusions");
  s = slide("Conclusions and next steps", "Conclusions and the follow-up work they motivate. The most immediate methodological improvement is to prevent metal atoms from being assigned to non-metals, which is the dominant failure mode when C and H outnumber the metal centre.");
  s.addText("Conclusions", { x: 0.6, y: 1.45, w: 5.9, h: 0.45, fontSize: 18, bold: true, color: INK, margin: 0, isTextBox: true, objectName: "conclusions heading" });
  bullets(s, [
    "Geometry is required for atom correspondence: FGW with a periodic neighbour kernel preserves local connectivity, whereas embeddings alone do not.",
    "Structure comparison across cell sizes requires an intensive representation: H/ΣZ, optimal transport, or the size-intensive FGW distance.",
    "MACE interaction-layer invariants are effectively element labels; for matching, the choice of layer is immaterial.",
    "MOF-off is chemically disjoint from OMat24; its closest OMat24 analogues are near-equilibrium AIMD frames.",
    "Rattled 300/500/1000 frames distort C–H bonds independently of their nominal temperature.",
  ], 0.6, 2.0, 5.9, 4.8, { size: 14, space: 10, name: "conclusions" });
  s.addText("Next steps", { x: 6.9, y: 1.45, w: 5.8, h: 0.45, fontSize: 18, bold: true, color: INK, margin: 0, isTextBox: true, objectName: "next steps heading" });
  bullets(s, [
    "Penalise metal-to-non-metal assignments, or weight metal centres, so that C and H cannot dominate the transport cost.",
    "Extend the matching benchmark to several hundred materials to resolve differences between UMA, MH1 and their combination.",
    "Score symmetry-equivalent assignments against the ideal Alexandria parent structures.",
    "Obtain access to the 468 restricted files; filter frames with interatomic distances below 0.8 Å throughout.",
  ], 6.9, 2.0, 5.8, 4.8, { size: 14, space: 10, name: "next steps" });

  s = slide("Data, code and deliverables", "Locations on the Della cluster. Every script in the analysis directory documents its inputs and outputs in its docstring.");
  table(s, ["Item", "Location"], [
    ["Embedding archive (read-only)", "/scratch/gpfs/ROSENGROUP/shared/aryan/saved_embeddings"],
    ["Source structures (LMDB, .traj)", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/data"],
    ["Analysis scripts", "/home/sv4956/materials/analysis"],
    ["Atom UMAPs; joint maps", "/home/sv4956/materials/runs/atom_umap_sr1m; runs/joint_umap"],
    ["MACE feature variants", "/home/sv4956/materials/data/mace_variants"],
    ["Matching benchmark and results", "/home/sv4956/materials/data/similar_structures{,_heldout}"],
    ["MOF-off vs OMat24 matrices", "/home/sv4956/materials/data/structure_fgw; data/grouped_fgw"],
    ["10M search, rerank results and CSVs", "/home/sv4956/materials/data/mof_search"],
    ["Interactive atom-matching viewer", "/home/sv4956/materials/viewer"],
  ], 0.6, 1.5, 12.1, [4.0, 8.1], { size: 14, name: "locations table" });
  caption(s, "CSVs: aimd_best_matches.csv (20 MOF–OMat24 pairs with explanations) and mof_neighbours_per_subset.csv (150 pairs: five nearest neighbours per MOF in AIMD 1000 K, AIMD 3000 K and rattled relax).", 0.6, 5.75, 12.1, 0.7);

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch(e => { console.error(e); process.exit(1); });
