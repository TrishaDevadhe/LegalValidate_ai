import io
import re
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extracts text from a .docx file using standard library zipfile and XML parsing.
    Extracts body paragraphs and table contents cleanly.
    """
    try:
        with io.BytesIO(file_bytes) as docx_file:
            with zipfile.ZipFile(docx_file) as z:
                xml_content = z.read("word/document.xml")
        
        tree = ET.fromstring(xml_content)
        # XML namespace for wordprocessingml
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        paragraphs = []
        for p in tree.iterfind('.//w:p', namespaces):
            texts = [node.text for node in p.iterfind('.//w:t', namespaces) if node.text]
            if texts:
                paragraphs.append("".join(texts))
                
        # Also extract table text if present
        for t in tree.iterfind('.//w:tbl', namespaces):
            for row in t.iterfind('.//w:tr', namespaces):
                row_texts = []
                for cell in row.iterfind('.//w:tc', namespaces):
                    cell_texts = [node.text for node in cell.iterfind('.//w:t', namespaces) if node.text]
                    if cell_texts:
                        row_texts.append("".join(cell_texts))
                if row_texts:
                    paragraphs.append(" | ".join(row_texts))
                    
        return "\n\n".join(paragraphs).strip()
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts text from a PDF byte stream with page metadata and OCR fallback.
    Returns:
        tuple: (full_text, pages_metadata)
        where pages_metadata is a list of dicts: [{'page': 1, 'text': '...'}]
    """
    pages_metadata = []
    full_text_list = []
    
    try:
        # 1. Native text extraction via pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    pages_metadata.append({"page": idx, "text": page_text})
                    full_text_list.append(f"--- [Page {idx}] ---\n{page_text}")
        
        extracted_text = "\n\n".join(full_text_list).strip()
        
        # 2. Check if text is suspiciously empty or scanned
        if len(extracted_text) < 50:
            try:
                doc = pdfium.PdfDocument(io.BytesIO(file_bytes))
                ocr_text_list = []
                pages_metadata = []
                
                for idx, page in enumerate(doc, start=1):
                    # Render page bitmap for OCR
                    bitmap = page.render(scale=2)
                    pil_img = bitmap.to_pil()
                    page_ocr = pytesseract.image_to_string(pil_img).strip()
                    if page_ocr:
                        pages_metadata.append({"page": idx, "text": page_ocr})
                        ocr_text_list.append(f"--- [Page {idx} (OCR)] ---\n{page_ocr}")
                
                ocr_full = "\n\n".join(ocr_text_list).strip()
                if len(ocr_full) > 0:
                    return ocr_full, pages_metadata
                else:
                    return "Warning: Extracted text is empty. Document appears blank or unscannable.", []
            except Exception as ocr_err:
                fallback_msg = extracted_text if extracted_text else "Document contains no readable text."
                return f"{fallback_msg}\n\n[OCR Fallback attempted: {str(ocr_err)}. Note: Ensure Tesseract OCR is installed on your host system for scanned image PDFs.]", pages_metadata

        return extracted_text, pages_metadata
    except Exception as e:
        return f"Error extracting PDF: {str(e)}", []


def extract_document_text(file_bytes: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Main document loader entry point supporting PDF, DOCX, and TXT files.
    """
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif ext in ['docx', 'doc']:
        text = extract_text_from_docx(file_bytes)
        return text, [{"page": 1, "text": text}]
    else:
        # Default text decode
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1', errors='ignore')
        return text, [{"page": 1, "text": text}]


def clean_text(text: str) -> str:
    """Normalizes whitespace and standardizes common quotes/dashes."""
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201C\u201D]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    # Remove repetitive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_text_by_sections(text: str, max_chars: int = 4000) -> List[str]:
    """
    Splits legal text into cohesive clauses and section chunks preserving clause boundaries.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Detect legal section headers, numbered clauses, or article dividers
    section_pattern = re.compile(
        r'(?=(?:^(?:SECTION|CLAUSE|ARTICLE|SCHEDULE|EXHIBIT|\d+\.\d+|\d+\.)\b)|(?:\n\n[A-Z0-9\s]{3,40}:))',
        re.IGNORECASE | re.MULTILINE
    )

    raw_sections = [s.strip() for s in section_pattern.split(text) if s and s.strip()]
    if not raw_sections or len(raw_sections) <= 1:
        # Fallback to paragraph splitting
        raw_sections = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sec in raw_sections:
        if current_length + len(sec) + 2 > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [sec]
            current_length = len(sec)
        else:
            current_chunk.append(sec)
            current_length += len(sec) + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def split_text_into_chunks_with_metadata(text: str, doc_id: str = "doc_1", max_chars: int = 3000) -> List[Dict[str, Any]]:
    """
    Splits text into chunks annotated with metadata: page, section, document_id, chunk_id.
    Ensures metadata survives the RAG and analysis pipeline.
    """
    sections = split_text_by_sections(text, max_chars=max_chars)
    chunks = []
    
    # Try detecting page markers if present
    current_page = 1
    for idx, sec in enumerate(sections, start=1):
        page_match = re.search(r'---\s*\[Page\s*(\d+)\]\s*---', sec)
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except ValueError:
                pass
                
        # Detect section title if first line resembles a heading
        first_line = sec.split('\n')[0].strip()
        section_name = first_line[:60] if len(first_line) < 80 else f"Section {idx}"
        
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{idx}",
            "document_id": doc_id,
            "page": current_page,
            "section": section_name,
            "text": sec
        })
        
    return chunks


def generate_pdf_report(final_state: dict) -> bytes:
    """
    Generates a professional multi-page PDF Audit Report for LegalValidate AI using ReportLab.
    Includes: Executive Summary, Document Details, Risk Analysis, Critic Verifications,
    Human Review Actions, and Legal Disclaimer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyTextCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#6B7280'),
        spaceBefore=12
    )

    story = []
    
    # 1. Header
    story.append(Paragraph("LegalValidate AI — Audit & Validation Report", title_style))
    story.append(Paragraph("Automated Multi-Agent Legal Review, RAG Grounding & Verification Trail", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E5E7EB'), spaceBefore=0, spaceAfter=12))
    
    # 2. Executive Metadata Box
    is_legal = final_state.get('is_legal', False)
    status_text = "VALID LEGAL CONTRACT" if is_legal else "NON-LEGAL / FLAGGED DOCUMENT"
    status_color = '#059669' if is_legal else '#DC2626'
    doc_type = final_state.get('document_type', 'Contract / Document')
    reason = final_state.get('classification_reason', 'Analysis completed.')
    
    # Count severities
    risks = final_state.get('risks', []) or []
    critic_reviews = final_state.get('critic_reviews', []) or []
    critic_map = {r.get('risk_id', ''): r for r in critic_reviews if isinstance(r, dict)}
    
    crit_count = sum(1 for r in risks if isinstance(r, dict) and r.get('severity') == 'CRITICAL')
    high_count = sum(1 for r in risks if isinstance(r, dict) and r.get('severity') == 'HIGH')
    med_count = sum(1 for r in risks if isinstance(r, dict) and r.get('severity') == 'MEDIUM')
    low_count = sum(1 for r in risks if isinstance(r, dict) and r.get('severity') == 'LOW')
    
    meta_table_data = [
        [Paragraph("<b>Document Classification:</b>", body_style), Paragraph(doc_type, body_style),
         Paragraph("<b>Validation Status:</b>", body_style), Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", body_style)],
        [Paragraph("<b>Identified Risks:</b>", body_style), Paragraph(f"Total: {len(risks)} | Critical: {crit_count} | High: {high_count} | Med: {med_count} | Low: {low_count}", body_style),
         Paragraph("<b>Classifier Note:</b>", body_style), Paragraph(reason, body_style)]
    ]
    
    t_meta = Table(meta_table_data, colWidths=[120, 180, 100, 132])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))
    
    # 3. Document Analysis Summary
    analysis = final_state.get('document_analysis', {}) or {}
    if analysis:
        story.append(Paragraph("Document Structure & Key Provisions", h1_style))
        struct_data = []
        if analysis.get('parties'):
            struct_data.append([Paragraph("<b>Parties:</b>", body_style), Paragraph(", ".join(analysis['parties']), body_style)])
        if analysis.get('effective_date'):
            struct_data.append([Paragraph("<b>Effective Date:</b>", body_style), Paragraph(str(analysis['effective_date']), body_style)])
        if analysis.get('term'):
            struct_data.append([Paragraph("<b>Term / Duration:</b>", body_style), Paragraph(str(analysis['term']), body_style)])
        if analysis.get('governing_law_and_jurisdiction'):
            struct_data.append([Paragraph("<b>Governing Law:</b>", body_style), Paragraph(str(analysis['governing_law_and_jurisdiction']), body_style)])
        if analysis.get('missing_sections'):
            struct_data.append([Paragraph("<b>Missing Safeguards:</b>", body_style), Paragraph("<font color='#DC2626'>" + ", ".join(analysis['missing_sections']) + "</font>", body_style)])
            
        if struct_data:
            t_struct = Table(struct_data, colWidths=[130, 402])
            t_struct.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
                ('PADDING', (0,0), (-1,-1), 5),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
            ]))
            story.append(t_struct)
            story.append(Spacer(1, 8))

    # 4. Plain-Language Executive Summary
    summary = final_state.get('summary', '')
    simplified = final_state.get('simplified_explanation', '')
    if summary or simplified:
        story.append(Paragraph("Executive Plain-Language Summary", h1_style))
        if summary:
            story.append(Paragraph(f"<b>Overview:</b> {summary}", body_style))
        if simplified:
            story.append(Paragraph(f"<b>Key Takeaways:</b> {simplified}", body_style))
        story.append(Spacer(1, 8))
        
    # 5. Risk Assessment & Critic Verification Table
    story.append(Paragraph("Risk Intelligence, RAG Grounding & Critic Audit", h1_style))
    
    if not risks:
        story.append(Paragraph("No critical legal risks or non-compliance vulnerabilities were detected.", body_style))
    else:
        # Human decision lookup
        human_decisions = {r.get('risk_id', ''): r for r in (final_state.get('human_reviews') or []) if isinstance(r, dict)}
        
        for idx, risk in enumerate(risks, start=1):
            if isinstance(risk, dict):
                r_id = risk.get('risk_id', f"risk_{idx}")
                clause = risk.get('clause', '')
                r_type = risk.get('risk_type', 'Contractual Risk')
                sev = risk.get('severity', 'MEDIUM')
                exp = risk.get('explanation', '')
                rec = risk.get('recommendation', '')
                evidence = risk.get('evidence', []) or []
                sources = risk.get('source_reference', []) or []
            else:
                r_id = f"risk_{idx}"
                clause = str(risk)
                r_type = "Identified Risk"
                sev = "MEDIUM"
                exp = str(risk)
                rec = "Review terms with legal counsel."
                evidence = []
                sources = []

            # Critic data
            c_review = critic_map.get(r_id, {})
            c_valid = c_review.get('is_valid', True)
            c_sev = c_review.get('verified_severity', sev)
            c_reason = c_review.get('reason', 'Audited by verification agent.')
            
            # Human Decision
            h_review = human_decisions.get(r_id, {})
            h_decision = h_review.get('decision', 'Pending Human Review')
            h_note = h_review.get('note', '')

            # Severity styling
            sev_colors = {'CRITICAL': '#7F1D1D', 'HIGH': '#B91C1C', 'MEDIUM': '#B45309', 'LOW': '#1E3A8A'}
            sev_c = sev_colors.get(c_sev, '#374151')

            risk_card_data = [
                [Paragraph(f"<b>#{idx}. {r_type}</b>", bold_body_style),
                 Paragraph(f"<font color='{sev_c}'><b>Severity: {c_sev}</b></font> (Critic: {'Valid' if c_valid else 'Rejected'})", body_style)],
                [Paragraph("<b>Target Clause:</b>", body_style), Paragraph(clause, body_style)],
                [Paragraph("<b>Risk Analysis:</b>", body_style), Paragraph(exp, body_style)],
                [Paragraph("<b>Recommendation:</b>", body_style), Paragraph(rec, body_style)],
            ]
            
            if sources and sources[0] != "Insufficient retrieved evidence":
                ref_text = "; ".join(sources[:2])
                risk_card_data.append([Paragraph("<b>RAG Baseline Reference:</b>", body_style), Paragraph(ref_text, body_style)])
                
            risk_card_data.append([
                Paragraph("<b>Critic Verdict / Reason:</b>", body_style),
                Paragraph(f"<b>{c_sev}</b> — {c_reason}", body_style)
            ])
            
            if h_decision != 'Pending Human Review' or h_note:
                risk_card_data.append([
                    Paragraph("<b>Human Reviewer Action:</b>", body_style),
                    Paragraph(f"Decision: <b>{h_decision}</b>" + (f" | Notes: {h_note}" if h_note else ""), body_style)
                ])

            t_card = Table(risk_card_data, colWidths=[120, 412])
            t_card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
                ('PADDING', (0,0), (-1,-1), 5),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
            ]))
            story.append(t_card)
            story.append(Spacer(1, 8))

    # 6. Legal Safety Disclaimer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceBefore=5, spaceAfter=8))
    story.append(Paragraph(
        "<b>LEGAL SAFETY DISCLAIMER:</b> LegalValidate AI is an automated machine-learning and analytical document validation assistant. "
        "This report and any accompanying findings do NOT constitute formal legal advice, an attorney-client relationship, or a substitute for review by qualified legal counsel. "
        "Users should always verify contract terms with authorized attorneys before executing binding agreements.",
        disclaimer_style
    ))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
