#!/usr/bin/env python3
"""Generate a test PDF with various table types and layouts for TPG testing."""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
import os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_medical_tables.pdf")

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(OUTPUT, pagesize=letter)
elements = []

# ─── Page 1: Title + simple body text + basic table ───
elements.append(Paragraph("Clinical Trial Report: Drug XR-7 Phase II Results", styles["Title"]))
elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph(
    "This report summarizes the Phase II clinical trial of Drug XR-7 conducted at "
    "Massachusetts General Hospital between January 2024 and September 2024. "
    "A total of 240 patients were enrolled across three treatment arms. "
    "The primary endpoint was reduction in systolic blood pressure at 12 weeks.",
    styles["BodyText"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Paragraph("Table 1: Patient Demographics", styles["Heading2"]))
demo_data = [
    ["Characteristic", "Drug XR-7\n(n=80)", "Placebo\n(n=80)", "Standard Care\n(n=80)"],
    ["Mean Age (years)", "58.3", "57.1", "59.0"],
    ["Female (%)", "52.5%", "48.8%", "51.3%"],
    ["BMI (kg/m2)", "28.4 +/- 3.2", "27.9 +/- 3.5", "28.1 +/- 3.0"],
    ["Diabetes (%)", "32.5%", "30.0%", "33.8%"],
    ["Hypertension Stage II (%)", "67.5%", "65.0%", "68.8%"],
    ["Prior CV Events (%)", "15.0%", "13.8%", "16.3%"],
]
t1 = Table(demo_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
t1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
elements.append(t1)

elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph(
    "Baseline characteristics were well-balanced across the three groups. "
    "No statistically significant differences were observed in age, gender distribution, "
    "or comorbidity prevalence (p > 0.05 for all comparisons).",
    styles["BodyText"]))

# ─── Page 2: Results table + adverse events ───
elements.append(PageBreak())
elements.append(Paragraph("Table 2: Primary and Secondary Endpoints at 12 Weeks", styles["Heading2"]))
results_data = [
    ["Endpoint", "Drug XR-7", "Placebo", "Standard Care", "p-value"],
    ["SBP reduction (mmHg)", "-18.4", "-4.2", "-9.1", "<0.001"],
    ["DBP reduction (mmHg)", "-11.2", "-2.8", "-5.6", "<0.001"],
    ["Heart rate (bpm)", "-3.1", "+0.5", "-1.2", "0.012"],
    ["HbA1c change (%)", "-0.4", "-0.1", "-0.2", "0.034"],
    ["LDL-C change (mg/dL)", "-12.3", "-2.1", "-5.8", "0.008"],
    ["eGFR change (mL/min)", "+2.1", "-0.3", "+0.8", "0.045"],
    ["Quality of Life Score", "+8.4", "+1.2", "+3.5", "<0.001"],
]
t2 = Table(results_data, colWidths=[1.8*inch, 1.1*inch, 1.1*inch, 1.3*inch, 0.9*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#d6eaf8")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
elements.append(t2)

elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph(
    "Drug XR-7 demonstrated statistically significant superiority over both placebo "
    "and standard care across all primary and secondary endpoints. The mean systolic "
    "blood pressure reduction of 18.4 mmHg exceeded the clinically meaningful threshold "
    "of 10 mmHg established by the FDA.",
    styles["BodyText"]))

elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph("Table 3: Adverse Events by System Organ Class", styles["Heading2"]))
ae_data = [
    ["System Organ Class", "Drug XR-7\nn (%)", "Placebo\nn (%)", "Severity"],
    ["Headache", "12 (15.0%)", "8 (10.0%)", "Mild"],
    ["Dizziness", "9 (11.3%)", "3 (3.8%)", "Mild-Moderate"],
    ["Nausea", "7 (8.8%)", "5 (6.3%)", "Mild"],
    ["Peripheral edema", "5 (6.3%)", "2 (2.5%)", "Mild"],
    ["Hypotension", "4 (5.0%)", "1 (1.3%)", "Moderate"],
    ["Elevated ALT", "3 (3.8%)", "2 (2.5%)", "Mild"],
    ["Bradycardia", "2 (2.5%)", "0 (0.0%)", "Moderate-Severe"],
    ["Syncope", "1 (1.3%)", "0 (0.0%)", "Severe"],
]
t3 = Table(ae_data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1.5*inch])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#922b21")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9ebea")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
]))
elements.append(t3)

elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph(
    "The most common adverse event was headache, reported in 15.0% of patients receiving "
    "Drug XR-7 compared to 10.0% in the placebo group. Two serious adverse events were "
    "recorded: one case of bradycardia requiring dose reduction and one episode of syncope "
    "leading to treatment discontinuation. No deaths were attributed to the study drug.",
    styles["BodyText"]))

# ─── Page 3: Nested/complex table + conclusions ───
elements.append(PageBreak())
elements.append(Paragraph("Table 4: Subgroup Analysis — SBP Reduction by Risk Category", styles["Heading2"]))
sub_data = [
    ["Subgroup", "n", "Drug XR-7\nSBP Change", "Placebo\nSBP Change", "Difference\n(95% CI)", "p-value"],
    ["Age < 60", "128", "-19.1", "-4.5", "-14.6 (-18.2, -11.0)", "<0.001"],
    ["Age >= 60", "112", "-17.6", "-3.8", "-13.8 (-17.8, -9.8)", "<0.001"],
    ["Male", "118", "-17.9", "-4.0", "-13.9 (-17.6, -10.2)", "<0.001"],
    ["Female", "122", "-18.8", "-4.4", "-14.4 (-18.1, -10.7)", "<0.001"],
    ["With Diabetes", "77", "-16.2", "-3.5", "-12.7 (-17.1, -8.3)", "<0.001"],
    ["Without Diabetes", "163", "-19.4", "-4.6", "-14.8 (-18.0, -11.6)", "<0.001"],
    ["BMI < 30", "148", "-18.8", "-4.3", "-14.5 (-17.9, -11.1)", "<0.001"],
    ["BMI >= 30", "92", "-17.5", "-3.9", "-13.6 (-18.0, -9.2)", "<0.001"],
]
t4 = Table(sub_data, colWidths=[1.3*inch, 0.4*inch, 1*inch, 1*inch, 1.6*inch, 0.8*inch])
t4.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#d6eaf8")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
elements.append(t4)

elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph(
    "Subgroup analyses demonstrated consistent efficacy of Drug XR-7 across all "
    "pre-specified subgroups. The treatment effect was maintained regardless of age, "
    "sex, diabetes status, or BMI category. Notably, patients without diabetes showed "
    "a numerically larger treatment effect (-14.8 mmHg vs -12.7 mmHg), though the "
    "interaction test was not statistically significant (p-interaction = 0.18).",
    styles["BodyText"]))

elements.append(Spacer(1, 0.3 * inch))
elements.append(Paragraph("Conclusions", styles["Heading1"]))
elements.append(Paragraph(
    "Drug XR-7 demonstrated robust and clinically meaningful reductions in blood pressure "
    "across a diverse patient population. The safety profile was acceptable, with most "
    "adverse events being mild to moderate in severity. These results support advancement "
    "to Phase III confirmatory trials. The Data Safety Monitoring Board has recommended "
    "continuation of the development program with expanded enrollment criteria.",
    styles["BodyText"]))

# Build PDF
doc.build(elements)
print(f"Generated: {OUTPUT}")
