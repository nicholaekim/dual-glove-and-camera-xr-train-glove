"""Generate the technical documentation PDF for the XR Trainer glove pipeline."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether, HRFlowable,
)

OUT = "XR_Trainer_Technical_Documentation.pdf"

# --- palette -----------------------------------------------------------------
INK = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#0b6e6e")
ACCENT_DK = colors.HexColor("#084f4f")
RULE = colors.HexColor("#c8d6d6")
CODE_BG = colors.HexColor("#f3f5f5")
CODE_BORDER = colors.HexColor("#d8e0e0")
TBL_HEAD = colors.HexColor("#0b6e6e")
TBL_ALT = colors.HexColor("#eef4f4")
MUTED = colors.HexColor("#5a6868")

# --- styles ------------------------------------------------------------------
ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


body = style("Body", fontName="Helvetica", fontSize=10, leading=15,
             textColor=INK, spaceAfter=8, alignment=TA_LEFT)
h1 = style("H1", fontName="Helvetica-Bold", fontSize=19, leading=23,
           textColor=ACCENT_DK, spaceBefore=4, spaceAfter=10)
h2 = style("H2", fontName="Helvetica-Bold", fontSize=14, leading=18,
           textColor=ACCENT_DK, spaceBefore=16, spaceAfter=6)
h3 = style("H3", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
           textColor=INK, spaceBefore=11, spaceAfter=4)
code = style("Code", fontName="Courier", fontSize=8.7, leading=12.5,
             textColor=colors.HexColor("#10302f"), backColor=CODE_BG,
             borderColor=CODE_BORDER, borderWidth=0.75, borderPadding=7,
             leftIndent=2, spaceBefore=10, spaceAfter=10)
mono = style("Mono", fontName="Courier", fontSize=9, leading=13, textColor=INK)
small = style("Small", fontName="Helvetica", fontSize=8.5, leading=12,
              textColor=MUTED)
cell = style("Cell", fontName="Helvetica", fontSize=8.6, leading=11.5,
             textColor=INK, spaceAfter=0)
cell_mono = style("CellMono", fontName="Courier", fontSize=8.2, leading=11,
                  textColor=INK, spaceAfter=0)
bullet = style("Bullet", parent=body, spaceAfter=3, leading=14)
cover_title = style("CoverTitle", fontName="Helvetica-Bold", fontSize=30,
                    leading=35, textColor=ACCENT_DK, spaceAfter=6)
cover_sub = style("CoverSub", fontName="Helvetica", fontSize=13, leading=18,
                  textColor=MUTED)


def code_block(text):
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace(" ", "&nbsp;").replace("\n", "<br/>"))
    return Paragraph(safe, code)


def bullets(items, st=bullet):
    return ListFlowable(
        [ListItem(Paragraph(t, st), leftIndent=12, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletColor=ACCENT,
    )


def _wrap_cells(data, st, header):
    """Wrap plain-string cells in Paragraphs so long text wraps inside its
    column instead of overflowing the page. Header row (if any) is left as a
    styled string so the table's bold/white header styling still applies."""
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = []
    for i, row in enumerate(data):
        if header and i == 0:
            out.append(row)
            continue
        out.append([Paragraph(esc(c), st) if isinstance(c, str) else c
                    for c in row])
    return out


def table(data, col_widths, header=True):
    data = _wrap_cells(data, cell, header)
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), TBL_HEAD),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.8),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ]
        for r in range(2, len(data), 2):
            cmds.append(("BACKGROUND", (0, r), (-1, r), TBL_ALT))
    t.setStyle(TableStyle(cmds))
    return t


def mono_table(data, col_widths):
    data = _wrap_cells(data, cell_mono, header=True)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), TBL_HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.6),
    ]
    for r in range(2, len(data), 2):
        cmds.append(("BACKGROUND", (0, r), (-1, r), TBL_ALT))
    t.setStyle(TableStyle(cmds))
    return t


def rule():
    return HRFlowable(width="100%", thickness=0.8, color=RULE,
                      spaceBefore=2, spaceAfter=10)


# --- page furniture ----------------------------------------------------------
DOC_TITLE = "XR Trainer / StretchSense Glove Pipeline"


def on_page(canvas, doc):
    canvas.saveState()
    w, h = letter
    if doc.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.6)
        canvas.line(0.9 * inch, h - 0.72 * inch, w - 0.9 * inch, h - 0.72 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.9 * inch, h - 0.66 * inch, DOC_TITLE)
        canvas.drawRightString(w - 0.9 * inch, h - 0.66 * inch,
                               "Technical Documentation")
        canvas.setStrokeColor(RULE)
        canvas.line(0.9 * inch, 0.62 * inch, w - 0.9 * inch, 0.62 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(w - 0.9 * inch, 0.45 * inch, "Page %d" % doc.page)
    canvas.restoreState()


story = []

# ---------------------------------------------------------------- COVER
story.append(Spacer(1, 1.7 * inch))
story.append(HRFlowable(width="38%", thickness=3, color=ACCENT, spaceAfter=18,
                        hAlign="LEFT"))
story.append(Paragraph("XR Trainer", cover_title))
story.append(Paragraph("StretchSense Glove Pipeline", cover_title))
story.append(Spacer(1, 10))
story.append(Paragraph("Technical Documentation", cover_sub))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Real-time hand-tracking capture, validation, 3D visualisation, "
    "recording, and data export.", cover_sub))
story.append(Spacer(1, 0.55 * inch))
story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceAfter=12))
cover_meta = Table([
    ["Package", "xr_hand  (version 0.1.0)"],
    ["Runtime", "Python 3.10 or newer"],
    ["Platform", "Windows / PowerShell (cross-platform Python core)"],
    ["Transport", "OSC over UDP, 127.0.0.1:9002"],
    ["Capture rate", "~60 Hz stream; configurable save rate"],
], colWidths=[1.5 * inch, 4.7 * inch])
cover_meta.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("TEXTCOLOR", (0, 0), (0, -1), ACCENT_DK),
    ("TEXTCOLOR", (1, 0), (1, -1), INK),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
]))
story.append(cover_meta)
story.append(PageBreak())

# ---------------------------------------------------------------- TOC
story.append(Paragraph("Contents", h1))
story.append(rule())
toc = [
    ("1", "System Overview", "How the components fit together"),
    ("2", "Architecture & Data Flow", "End-to-end pipeline stages"),
    ("3", "Installation", "Environment setup on Windows"),
    ("4", "Quick Start", "Common workflows in one place"),
    ("5", "Command Reference", "Every script and its options"),
    ("6", "The Joint Model", "26-joint layout and skeleton"),
    ("7", "Wire Format", "OSC packet structure on the network"),
    ("8", "Module Reference", "What each source file does"),
    ("9", "Data Formats", "JSONL, CSV, and Excel outputs"),
    ("10", "Validation & Stream Health", "Error and warning semantics"),
    ("11", "Mock Mode & Fault Injection", "Running without hardware"),
    ("12", "Testing", "Automated and manual checks"),
    ("13", "Troubleshooting", "Common problems and fixes"),
]
rows = [["", "Section", "Description"]]
for num, title, desc in toc:
    rows.append([num, title, desc])
story.append(table(rows, [0.45 * inch, 2.4 * inch, 3.4 * inch]))
story.append(PageBreak())

# ---------------------------------------------------------------- 1 OVERVIEW
story.append(Paragraph("1. System Overview", h1))
story.append(rule())
story.append(Paragraph(
    "This project is a real-time pipeline for capturing, inspecting, and "
    "exporting hand-tracking data from a StretchSense Reality Glove driven by "
    "the XR Trainer application. The glove streams skeletal pose data over the "
    "network; the pipeline receives it, checks it for integrity, reconstructs "
    "the hand in 3D, and lets you record sessions for later review and "
    "analysis.", body))
story.append(Paragraph(
    "Each hand is described by 26 joints following the OpenXR "
    "<font name='Courier'>XR_HAND_JOINT</font> layout. Every joint carries a "
    "local position and an orientation quaternion, updated roughly 60 times a "
    "second.", body))

story.append(Paragraph("What you can do with it", h3))
story.append(bullets([
    "<b>Watch the hand live</b> in an interactive 3D viewer while the glove streams.",
    "<b>Record sessions</b> to disk and scrub through them later like a video.",
    "<b>Validate the stream</b> for dropped, frozen, malformed, or NaN-laden packets.",
    "<b>Export</b> any time window to a tidy, formatted spreadsheet, or a full session to CSV.",
    "<b>Develop without hardware</b> using a built-in mock generator with injectable faults.",
]))

story.append(Paragraph("Design principles", h3))
story.append(bullets([
    "<b>Numeric-only downstream:</b> string header fields are normalised to "
    "placeholders at the receiver, so the validator and parser only ever see numbers.",
    "<b>Thread isolation:</b> the network server runs on its own thread and hands "
    "packets to the viewer through a bounded, thread-safe queue.",
    "<b>Swappable rendering:</b> the viewer is the only surface that touches "
    "matplotlib, so the backend can be replaced without changing the scripts.",
    "<b>One canonical wire layout:</b> the mock generator emits exactly the same "
    "187-value payload the real glove does, so the whole pipeline is exercised identically.",
]))
story.append(PageBreak())

# ---------------------------------------------------------------- 2 ARCHITECTURE
story.append(Paragraph("2. Architecture & Data Flow", h1))
story.append(rule())
story.append(Paragraph(
    "Data moves through the pipeline in a fixed sequence of stages. Each stage "
    "has a single responsibility and a narrow interface to the next.", body))
story.append(Spacer(1, 4))
story.append(code_block(
    "glove / XR Trainer\n"
    "      |  OSC over UDP  (127.0.0.1:9002)\n"
    "      v\n"
    "  [ receiver ]   bind socket, normalise strings, queue raw values\n"
    "      |\n"
    "      v\n"
    "  [ validator ]  length / NaN / zero / quaternion checks per packet\n"
    "      |          StreamMonitor tracks counter freezes & gaps\n"
    "      v\n"
    "  [ parser ]     187 values  ->  HandFrame (26 Joint records)\n"
    "      |\n"
    "      +---------------------------+\n"
    "      v                           v\n"
    "  [ kinematics ]             [ recorder ]\n"
    "  local -> world FK          append frame to JSONL\n"
    "      |                           |\n"
    "      v                           v\n"
    "  [ viz3d / playback ]       recordings/*.jsonl\n"
    "  interactive 3D viewer            |\n"
    "                                   v\n"
    "                       [ export ]  ->  CSV / XLSX"))

story.append(Paragraph("Stage responsibilities", h3))
story.append(table([
    ["Stage", "Module", "Responsibility"],
    ["Receive", "receiver.py", "Bind the UDP socket, dispatch by OSC address, queue raw numeric payloads."],
    ["Validate", "validator.py", "Reject structurally broken packets; warn on suspicious values and counter behaviour."],
    ["Parse", "parser.py", "Turn a flat value list into a typed HandFrame with 26 named joints."],
    ["Kinematics", "kinematics.py", "Chain parent-relative transforms into world-space joint positions."],
    ["Visualise", "viz3d.py / playback_viewer.py", "Draw the skeleton live, or scrub a recording with a data panel."],
    ["Record", "recorder.py", "Write validated frames to JSONL, optionally downsampled."],
    ["Export", "export.py / export_xlsx.py", "Convert recordings to CSV or a formatted Excel workbook."],
], [0.85 * inch, 1.6 * inch, 3.8 * inch]))
story.append(PageBreak())

# ---------------------------------------------------------------- 3 INSTALL
story.append(Paragraph("3. Installation", h1))
story.append(rule())
story.append(Paragraph("Requirements", h3))
story.append(bullets([
    "Python 3.10 or newer.",
    "The XR Trainer application and a StretchSense Reality Glove for live capture "
    "(not required for mock mode).",
]))
story.append(Paragraph("Dependencies", h3))
story.append(Paragraph(
    "Installed automatically by the step below. Core runtime packages:", body))
story.append(table([
    ["Package", "Purpose"],
    ["numpy", "Vector / matrix maths for forward kinematics."],
    ["matplotlib", "Interactive 3D viewer and playback UI."],
    ["python-osc", "OSC server that receives UDP packets from XR Trainer."],
    ["openpyxl", "Writes the formatted Excel export."],
    ["pytest (dev)", "Test runner for the automated suite."],
], [1.5 * inch, 4.7 * inch]))

story.append(Paragraph("Setup (Windows / PowerShell)", h3))
story.append(code_block(
    'cd "C:\\Users\\nkim2\\Desktop\\xr trainer"\n'
    "python -m venv .venv\n"
    ".\\.venv\\Scripts\\Activate.ps1\n"
    'pip install -e ".[dev]"'))
story.append(Paragraph(
    "Installing with <font name='Courier'>-e</font> puts the package in editable "
    "mode, so source edits take effect without reinstalling.", body))
story.append(Paragraph("If activation is blocked", h3))
story.append(Paragraph(
    "If <font name='Courier'>Activate.ps1</font> is blocked by execution policy, "
    "run this once per machine, then re-open the shell:", body))
story.append(code_block("Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"))
story.append(PageBreak())

# ---------------------------------------------------------------- 4 QUICK START
story.append(Paragraph("4. Quick Start", h1))
story.append(rule())
story.append(Paragraph(
    "With XR Trainer running and the glove streaming to "
    "<font name='Courier'>127.0.0.1:9002</font>, the workflows below cover most "
    "day-to-day use.", body))

story.append(Paragraph("One-command record + playback + export", h3))
story.append(Paragraph(
    "The recommended path. Records a clip, opens it for review, then writes a CSV:", body))
story.append(code_block(
    ".\\scripts\\record_and_export.ps1                # 10 s default, 5 Hz\n"
    ".\\scripts\\record_and_export.ps1 -Duration 30   # record 30 s\n"
    ".\\scripts\\record_and_export.ps1 -Hz 10         # save 10 frames/sec"))
story.append(Paragraph(
    "Produces <font name='Courier'>recordings\\realglove_YYYYMMDD_HHMMSS.jsonl</font> "
    "and a matching <font name='Courier'>.csv</font>.", body))

story.append(Paragraph("Record a session manually", h3))
story.append(code_block("python scripts/record.py"))
story.append(Paragraph(
    "Starts immediately and runs until you press Ctrl+C, saving to "
    "<font name='Courier'>recordings\\session_&lt;id&gt;.jsonl</font> and printing "
    "the playback command. Add <font name='Courier'>--mock</font> to record "
    "without the glove.", body))

story.append(Paragraph("Scrub a recording like a video", h3))
story.append(code_block("python scripts/playback.py recordings\\session_<id>.jsonl"))
story.append(Paragraph(
    "Opens the 3D hand on the left and a per-joint data panel on the right. "
    "Full controls are described in section 5.", body))

story.append(Paragraph("Live viewer only", h3))
story.append(code_block(
    "python scripts/run_osc.py                       # live 3D viewer\n"
    "python scripts/run_osc.py --record recordings/my.jsonl   # view + record\n"
    "python scripts/run_osc.py --dump --no-viz       # diagnose stream, no GUI"))

story.append(Paragraph("Without the glove (mock)", h3))
story.append(code_block(
    "python scripts/run_mock.py                      # animated mock viewer\n"
    "python scripts/record.py --mock --duration 10   # mock recording\n"
    "python scripts/test_faults.py                   # headless fault checks"))
story.append(PageBreak())

# ---------------------------------------------------------------- 5 COMMAND REF
story.append(Paragraph("5. Command Reference", h1))
story.append(rule())

story.append(Paragraph("run_osc.py — live pipeline", h2))
story.append(Paragraph(
    "Receiver, validator, parser, and 3D viewer wired together for the real glove.", body))
story.append(mono_table([
    ["Option", "Default", "Meaning"],
    ["--host", "127.0.0.1", "Interface to bind (0.0.0.0 accepts any)"],
    ["--port", "9002", "UDP port XR Trainer sends to"],
    ["--address", "/v1/animation/", "OSC address of the kinematic stream"],
    ["--dump", "off", "Log one line per received packet"],
    ["--no-viz", "off", "Validate only, no GUI window"],
    ["--record PATH", "none", "Also write each frame to a .jsonl file"],
    ["--duration S", "none", "Auto-close the viewer after S seconds"],
    ["--hz N", "5.0", "Frames saved per second (0 = keep all)"],
], [1.25 * inch, 1.35 * inch, 3.6 * inch]))

story.append(Paragraph("record.py — headless recorder", h2))
story.append(Paragraph(
    "Writes validated frames to JSONL with no viewer. Supports both live OSC and "
    "mock sources.", body))
story.append(mono_table([
    ["Option", "Default", "Meaning"],
    ["--mock", "off", "Use the mock generator instead of OSC"],
    ["--port", "9002", "UDP port for live capture"],
    ["--output PATH", "auto", "Output file (default recordings/session_*)"],
    ["--duration S", "none", "Stop after S seconds (else Ctrl+C)"],
    ["--hz N", "5.0", "Frames saved per second (0 = keep all)"],
], [1.25 * inch, 1.35 * inch, 3.6 * inch]))

story.append(Paragraph("playback.py — scrubbable viewer", h2))
story.append(Paragraph("Loads a whole recording and lets you move through it.", body))
story.append(table([
    ["Control", "Action"],
    ["Frame slider / arrow keys", "Step through the clip one frame at a time."],
    ["Select (range) slider", "Pick a time window for playback and export."],
    ["Play / Pause", "Auto-advance within the selected window."],
    ["Export selection -> Excel", "Write just the selected frames to a formatted .xlsx."],
], [2.1 * inch, 4.1 * inch]))
story.append(Paragraph(
    "The data panel shows both hands' world-space joint positions and highlights "
    "the joints that moved into the current frame.", body))

story.append(Paragraph("export.py — JSONL to CSV", h2))
story.append(mono_table([
    ["Option", "Default", "Meaning"],
    ["input", "required", "Path to a .jsonl recording"],
    ["--output PATH", "same name", "Output .csv path"],
    ["--hand left|right", "both", "Filter to a single hand"],
], [1.45 * inch, 1.15 * inch, 3.6 * inch]))

story.append(Paragraph("Helper scripts", h2))
story.append(table([
    ["Script", "Purpose"],
    ["run_mock.py", "Full mock pipeline with an animated viewer; edit toggles to inject faults."],
    ["test_faults.py", "Headless check that each injected fault is detected as expected."],
    ["record_and_export.ps1", "One-command record + playback + export wrapper."],
], [1.7 * inch, 4.5 * inch]))
story.append(PageBreak())

# ---------------------------------------------------------------- 6 JOINT MODEL
story.append(Paragraph("6. The Joint Model", h1))
story.append(rule())
story.append(Paragraph(
    "Each hand has 26 joints, indexed 0-25 in the OpenXR order. The wrist (index "
    "1) is the root of the skeleton; the palm and the five finger chains hang off "
    "it. Bone connectivity is defined as parent-child index pairs used both for "
    "drawing and for forward kinematics.", body))

story.append(Paragraph("Joint indices", h3))
story.append(mono_table([
    ["Idx", "Joint", "Idx", "Joint"],
    ["0", "PALM", "13", "MIDDLE_INTERMEDIATE"],
    ["1", "WRIST", "14", "MIDDLE_DISTAL"],
    ["2", "THUMB_METACARPAL", "15", "MIDDLE_TIP"],
    ["3", "THUMB_PROXIMAL", "16", "RING_METACARPAL"],
    ["4", "THUMB_DISTAL", "17", "RING_PROXIMAL"],
    ["5", "THUMB_TIP", "18", "RING_INTERMEDIATE"],
    ["6", "INDEX_METACARPAL", "19", "RING_DISTAL"],
    ["7", "INDEX_PROXIMAL", "20", "RING_TIP"],
    ["8", "INDEX_INTERMEDIATE", "21", "LITTLE_METACARPAL"],
    ["9", "INDEX_DISTAL", "22", "LITTLE_PROXIMAL"],
    ["10", "INDEX_TIP", "23", "LITTLE_INTERMEDIATE"],
    ["11", "MIDDLE_METACARPAL", "24", "LITTLE_DISTAL"],
    ["12", "MIDDLE_PROXIMAL", "25", "LITTLE_TIP"],
], [0.5 * inch, 2.55 * inch, 0.5 * inch, 2.55 * inch]))

story.append(Paragraph("Skeleton hierarchy", h3))
story.append(Paragraph(
    "The wrist connects to the palm and to each finger's first joint. Every finger "
    "is a chain from metacarpal to tip:", body))
story.append(code_block(
    "WRIST -> PALM\n"
    "WRIST -> THUMB_METACARPAL  -> PROXIMAL -> DISTAL -> TIP\n"
    "WRIST -> INDEX_METACARPAL  -> PROXIMAL -> INTERMEDIATE -> DISTAL -> TIP\n"
    "WRIST -> MIDDLE_METACARPAL -> PROXIMAL -> INTERMEDIATE -> DISTAL -> TIP\n"
    "WRIST -> RING_METACARPAL   -> PROXIMAL -> INTERMEDIATE -> DISTAL -> TIP\n"
    "WRIST -> LITTLE_METACARPAL -> PROXIMAL -> INTERMEDIATE -> DISTAL -> TIP"))

story.append(Paragraph("Core data structures", h3))
story.append(Paragraph(
    "Defined in <font name='Courier'>joints.py</font>. A "
    "<font name='Courier'>Joint</font> holds a name, a position, and an XYZW "
    "quaternion. A <font name='Courier'>HandFrame</font> bundles 26 joints with "
    "packet metadata.", body))
story.append(code_block(
    "Joint(name, x, y, z, qw, qx, qy, qz)\n\n"
    "HandFrame(\n"
    "    timestamp,        # tick counter, used time-like\n"
    "    packet_counter,   # for freeze / gap detection\n"
    "    hand_side,        # 'left' or 'right'\n"
    "    frame_id,\n"
    "    status,\n"
    "    joints,           # list of 26 Joint records\n"
    ")"))
story.append(PageBreak())

# ---------------------------------------------------------------- 7 WIRE FORMAT
story.append(Paragraph("7. Wire Format", h1))
story.append(rule())
story.append(Paragraph(
    "XR Trainer sends to <font name='Courier'>127.0.0.1:9002</font> on the OSC "
    "address <font name='Courier'>/v1/animation/kinematic/all</font> with exactly "
    "187 arguments per packet.", body))

story.append(Paragraph("Packet layout (187 values)", h3))
story.append(table([
    ["Range", "Count", "Contents"],
    ["[0]", "1", "Tick counter (int) — used as packet_counter and a time-like value."],
    ["[1]", "1", "Frame id (int)."],
    ["[2]", "1", "Status / hand code (int)."],
    ["[3]", "1", "Device-label string, e.g. \"Reality Glove (L)\" — encodes hand side."],
    ["[4]", "1", "Device-family string, e.g. \"Reality Glove\"."],
    ["[5..186]", "182", "26 joints x 7 values: x, y, z, qx, qy, qz, qw."],
], [0.95 * inch, 0.6 * inch, 4.65 * inch]))
story.append(Paragraph("Header total: 5 values. Joints: 26 x 7 = 182. Grand total: 187.", small))

story.append(Paragraph("Per-joint value order", h3))
story.append(Paragraph(
    "Each joint contributes seven floats in this order. Position is "
    "<b>parent-relative</b> (local space); the quaternion is <b>XYZW</b> with unit "
    "magnitude.", body))
story.append(code_block("x   y   z   qx   qy   qz   qw"))

story.append(Paragraph("Hand-side resolution", h3))
story.append(Paragraph(
    "The hand is determined from the device-label string at argument [3] "
    "(<font name='Courier'>(L)</font> or <font name='Courier'>(R)</font>, or a "
    "trailing <font name='Courier'>L</font>/<font name='Courier'>R</font>). The "
    "receiver then replaces both string header fields with "
    "<font name='Courier'>0.0</font> placeholders so the validator and parser "
    "only handle numeric data.", body))

story.append(Paragraph("Mock / legacy addresses", h3))
story.append(Paragraph(
    "The receiver also accepts <font name='Courier'>/hand/left</font> and "
    "<font name='Courier'>/hand/right</font> with 187 numeric arguments, where the "
    "hand side comes from the address rather than a label string. This lets the "
    "mock sender drive the same receiver as the real glove.", body))
story.append(PageBreak())

# ---------------------------------------------------------------- 8 MODULES
story.append(Paragraph("8. Module Reference", h1))
story.append(rule())
story.append(Paragraph("Package <font name='Courier'>src/xr_hand/</font>", h3))
story.append(table([
    ["Module", "Responsibility"],
    ["joints.py", "26 joint names, bone connectivity, and the Joint / HandFrame dataclasses."],
    ["parser.py", "Convert a 187-value payload into a HandFrame; raises PacketParseError on bad input."],
    ["validator.py", "Per-packet structural checks plus StreamMonitor for cross-packet counter health."],
    ["receiver.py", "Threaded OSC/UDP server; dispatches by address and queues numeric payloads."],
    ["kinematics.py", "quat_to_mat3 and forward_kinematics: local transforms to world positions."],
    ["mock.py", "MockHandGenerator producing canonical payloads with optional injected faults."],
    ["recorder.py", "FrameRecorder: write/load JSONL, with optional per-hand downsampling."],
    ["viz3d.py", "HandViewer: live matplotlib 3D skeleton with palm fill and fingertip markers."],
    ["playback_viewer.py", "PlaybackViewer: scrubbable 3D + data panel + window selection and Excel export."],
    ["export_xlsx.py", "write_workbook: tidy, styled Excel output with summary and joint-data sheets."],
], [1.5 * inch, 4.7 * inch]))

story.append(Paragraph("Scripts <font name='Courier'>scripts/</font>", h3))
story.append(table([
    ["Script", "Responsibility"],
    ["run_osc.py", "Live real-glove pipeline with optional recording and auto-close."],
    ["run_mock.py", "Live in-process mock pipeline; edit toggles to inject faults."],
    ["record.py", "Headless recorder for live or mock sources."],
    ["playback.py", "Open and scrub a recorded .jsonl session."],
    ["export.py", "Convert a .jsonl recording to wide CSV."],
    ["test_faults.py", "Headless fault-injection sanity check."],
    ["record_and_export.ps1", "One-command record + playback + export pipeline."],
], [1.7 * inch, 4.5 * inch]))

story.append(Paragraph("Key interfaces", h3))
story.append(Paragraph(
    "The receiver exposes a simple producer/consumer surface; the viewer drains "
    "the queue each animation tick:", body))
story.append(code_block(
    "rx = OSCHandReceiver(host, port, kinematic_addr=...)\n"
    "rx.start()\n"
    "for hand, raw in rx.drain(max_items=16):\n"
    "    result = validate_raw_message(raw)\n"
    "    if result.is_valid:\n"
    "        frame = parse_hand_message(raw, hand_side_hint=hand)\n"
    "        world = forward_kinematics(frame)\n"
    "rx.stop()"))
story.append(PageBreak())

# ---------------------------------------------------------------- 9 DATA FORMATS
story.append(Paragraph("9. Data Formats", h1))
story.append(rule())

story.append(Paragraph("JSONL recording", h2))
story.append(Paragraph(
    "One JSON object per line — easy to inspect, grep, or stream. Each line is one "
    "hand frame:", body))
story.append(code_block(
    "{\n"
    '  "wall_time": 1748123456.789,   # real clock time, for replay timing\n'
    '  "timestamp": 4.233,            # OSC tick counter\n'
    '  "packet_counter": 254,\n'
    '  "hand_side": "right",\n'
    '  "frame_id": 254,\n'
    '  "status": 0,\n'
    '  "joints": [\n'
    '    {"name": "PALM", "x": 0.0, "y": 0.04, "z": 0.0,\n'
    '     "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},\n'
    "    ...\n"
    "  ]\n"
    "}"))

story.append(Paragraph("Sample rate", h3))
story.append(Paragraph(
    "Recordings are downsampled to 5 frames/sec by default "
    "(<font name='Courier'>--hz 5</font>). The glove still streams at ~60 Hz; only "
    "the saved frames are throttled, and each hand is throttled independently so "
    "both keep the full target rate. Every saved frame keeps its exact "
    "<font name='Courier'>wall_time</font> and <font name='Courier'>timestamp</font>. "
    "Pass <font name='Courier'>--hz 0</font> to keep every frame.", body))

story.append(Paragraph("CSV export", h2))
story.append(Paragraph(
    "One row per frame, wide layout. Positions and quaternions are kept in the "
    "device's native local frame; forward kinematics is left to the analyst. "
    "Columns:", body))
story.append(code_block(
    "wall_time, iso_time, timestamp, packet_counter,\n"
    "hand_side, frame_id, status,\n"
    "PALM_x, PALM_y, PALM_z, PALM_qw, PALM_qx, PALM_qy, PALM_qz,\n"
    "WRIST_x, ...  (26 joints x 7 columns = 182 columns)"))

story.append(Paragraph("Excel export", h2))
story.append(Paragraph(
    "Written from the playback viewer's selected window. Unlike the wide CSV, the "
    "workbook is a tidy long table — one row per frame x hand x joint — with two "
    "sheets:", body))
story.append(bullets([
    "<b>Summary</b>: source file, the selected window in elapsed and clock time, "
    "frame count, joints per hand, and the coordinate space used.",
    "<b>Joint data</b>: frame, elapsed seconds, clock, hand, joint, world XYZ, "
    "rotation quaternion (qw, qx, qy, qz), and a <i>moved</i> flag.",
]))
story.append(Paragraph(
    "The sheet has a frozen, filterable header, per-hand row colouring (pale cyan "
    "for left, pale red for right), and bold joint names on rows that moved. World "
    "positions are used here because they move with the hand, alongside the raw "
    "quaternion that drives the motion.", body))
story.append(PageBreak())

# ---------------------------------------------------------------- 10 VALIDATION
story.append(Paragraph("10. Validation & Stream Health", h1))
story.append(rule())
story.append(Paragraph(
    "Validation has two layers: per-packet structural checks, and cross-packet "
    "stream monitoring. Errors mean a packet is unusable and is dropped; warnings "
    "mean it is usable but something looks off and is logged.", body))

story.append(Paragraph("Per-packet checks", h3))
story.append(table([
    ["Check", "Severity", "Condition"],
    ["Length", "Error", "Argument count is not exactly 187."],
    ["Non-finite", "Error", "Any value is NaN or infinite."],
    ["All-zero joint", "Warning", "A joint's seven values are all ~0."],
    ["Quaternion magnitude", "Warning", "|q| deviates from 1.0 by more than 0.1."],
], [1.85 * inch, 0.85 * inch, 3.5 * inch]))

story.append(Paragraph("Stream monitor (per hand)", h3))
story.append(Paragraph(
    "Real-device tick counters legitimately repeat and skip within batched "
    "emissions, so only <i>persistent</i> freezes and <i>large</i> gaps are "
    "treated as problems. Tracked per hand by "
    "<font name='Courier'>StreamMonitor</font>:", body))
story.append(table([
    ["Condition", "Default threshold", "Meaning"],
    ["Counter freeze", "20+ identical in a row", "Stream likely stalled or stuck."],
    ["Counter gap", "10+ ticks skipped", "Packets dropped in transit."],
    ["Counter backwards", "any decrease", "Out-of-order or reset stream."],
], [1.55 * inch, 1.75 * inch, 2.9 * inch]))
story.append(Paragraph(
    "Defaults assume roughly a 100 Hz packet rate; adjust the thresholds if your "
    "stream rate differs substantially. A freeze emits one warning when it starts "
    "and one when the counter resumes, rather than spamming every packet.", body))
story.append(PageBreak())

# ---------------------------------------------------------------- 11 MOCK
story.append(Paragraph("11. Mock Mode & Fault Injection", h1))
story.append(rule())
story.append(Paragraph(
    "<font name='Courier'>MockHandGenerator</font> emits the same 187-value "
    "payload the real glove does, so the entire pipeline runs identically without "
    "hardware. The hand animates through an open-to-fist curl driven by a slow "
    "sine. Each generator drives one hand; the two are offset along X so they do "
    "not overlap.", body))

story.append(Paragraph("Fault toggles", h3))
story.append(Paragraph(
    "Set these attributes on a generator to inject specific failure modes and "
    "confirm the validator catches them:", body))
story.append(table([
    ["Toggle", "Effect", "Expected detection"],
    ["stuck_fingertip = NAME", "One tip stops moving (frozen offset).", "None — visual-only, stream stays valid."],
    ["freeze_counter = True", "Packet counter never increments.", "StreamMonitor freeze warning."],
    ["zero_joint = NAME", "One joint becomes all zeros.", "All-zero joint warning."],
    ["wrong_length = True", "Payload truncated by one value.", "Length-mismatch error."],
    ["random_missing_prob = p", "Each value has p chance of being NaN.", "Non-finite error."],
], [1.75 * inch, 2.25 * inch, 2.2 * inch]))

story.append(Paragraph("Trying it interactively", h3))
story.append(Paragraph(
    "Edit the fault block near the top of "
    "<font name='Courier'>run_mock.py</font> and run it to watch a fault in the "
    "live viewer:", body))
story.append(code_block(
    "right_gen.stuck_fingertip = \"INDEX_TIP\"\n"
    "# right_gen.freeze_counter  = True\n"
    "# right_gen.zero_joint      = \"MIDDLE_PROXIMAL\"\n"
    "# right_gen.wrong_length    = True\n"
    "# right_gen.random_missing_prob = 0.005"))
story.append(PageBreak())

# ---------------------------------------------------------------- 12 TESTING
story.append(Paragraph("12. Testing", h1))
story.append(rule())
story.append(Paragraph("Automated suite", h3))
story.append(Paragraph("Run the full test suite with pytest:", body))
story.append(code_block("pytest -q"))
story.append(table([
    ["Test file", "Covers"],
    ["test_pipeline.py", "Parser correctness, per-packet validation, and StreamMonitor behaviour."],
    ["test_recorder.py", "JSONL record/load round-trip fidelity."],
], [1.7 * inch, 4.5 * inch]))

story.append(Paragraph("Fault-injection check", h3))
story.append(Paragraph(
    "A headless script runs the mock generator through the validator for each "
    "fault and reports whether the expected error or warning fired:", body))
story.append(code_block("python scripts/test_faults.py"))
story.append(Paragraph(
    "It verifies a clean stream stays clean, a stuck fingertip stays valid, and "
    "each injected fault (frozen counter, zero joint, wrong length, random NaN) is "
    "detected. It prints a per-check PASS/FAIL line and a final tally.", body))

story.append(Paragraph("Manual verification", h3))
story.append(Paragraph(
    "When first connecting XR Trainer, confirm the wire format before relying on "
    "the viewer:", body))
story.append(code_block("python scripts/run_osc.py --dump --no-viz"))
story.append(Paragraph(
    "This logs one line per packet so you can confirm the address pattern and "
    "argument count (187) match what the validator expects.", body))
story.append(PageBreak())

# ---------------------------------------------------------------- 13 TROUBLESHOOTING
story.append(Paragraph("13. Troubleshooting", h1))
story.append(rule())
story.append(table([
    ["Symptom", "Likely cause & fix"],
    ["No packets / empty viewer",
     "Confirm XR Trainer is streaming to 127.0.0.1:9002. Run with --dump --no-viz to "
     "see whether any OSC packets arrive at all."],
    ["\"unexpected arg count\" warnings",
     "The stream is not sending 187 arguments on the expected address. Check the "
     "--address value and the XR Trainer output format."],
    ["\"could not determine hand\" skips",
     "The device-label string at arg [3] does not contain (L)/(R) or a trailing "
     "L/R. Verify the glove label in XR Trainer."],
    ["Counter-frozen warnings",
     "The stream stalled (20+ identical counters). Check the glove connection and "
     "XR Trainer status; a brief freeze that resumes is logged and is usually fine."],
    ["Activate.ps1 is blocked",
     "Run Set-ExecutionPolicy -Scope CurrentUser RemoteSigned once, then reopen the "
     "shell."],
    ["Quaternion-magnitude warnings",
     "Joint orientation deviates from unit length by more than 0.1 — usually a "
     "transient; persistent cases suggest a data issue upstream."],
    ["Recording grows slowly",
     "Frames are downsampled to 5 Hz by default. Pass --hz 0 to keep every frame, "
     "or a higher value for more detail."],
    ["Excel export looks empty",
     "Make sure a window is selected on the range slider before clicking Export; "
     "only the selected frames are written."],
], [1.85 * inch, 4.35 * inch]))

story.append(Spacer(1, 16))
story.append(rule())
story.append(Paragraph(
    "End of document. For the authoritative behaviour of any component, consult "
    "the corresponding module in <font name='Courier'>src/xr_hand/</font>.", small))


# --- build -------------------------------------------------------------------
def build():
    doc = BaseDocTemplate(
        OUT, pagesize=letter,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.95 * inch, bottomMargin=0.8 * inch,
        title="XR Trainer / StretchSense Glove Pipeline — Technical Documentation",
        author="XR Trainer",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=on_page)])
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
