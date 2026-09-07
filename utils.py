import pdfplumber
import pypdfium2 as pdfium
import pytesseract
import io

def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file byte stream using pdfplumber, with OCR fallback."""
    try:
        text = ""
        # 1. Try extracting text via pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        text = text.strip()
        
        # 2. Check if extracted text is empty or near-empty (threshold: e.g. < 50 characters)
        if len(text) < 50:
            # Trigger OCR fallback
            try:
                doc = pdfium.PdfDocument(io.BytesIO(file_bytes))
                ocr_text = ""
                for page in doc:
                    # Render page to bitmap at scale=2 for decent OCR accuracy
                    bitmap = page.render(scale=2)
                    pil_img = bitmap.to_pil()
                    ocr_text += pytesseract.image_to_string(pil_img) + "\n"
                
                ocr_text = ocr_text.strip()
                if len(ocr_text) > 0:
                    return ocr_text
                else:
                    return "Warning: Extracted text is empty and OCR could not extract any text from the scanned document."
            except Exception as ocr_err:
                # If OCR fails due to missing dependencies (e.g. Tesseract not installed on Windows),
                # return a helpful message instead of crashing.
                return f"{text}\n\n[OCR Fallback attempted but failed: {str(ocr_err)}. Ensure Tesseract is installed and in your system PATH for scanned PDF support.]"
        
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def clean_text(text):
    """Basic text cleaning."""
    return " ".join(text.split())

def split_text_by_sections(text, max_chars=30000):
    """
    Splits text into logical sections based on headings or clause starts,
    ensuring each chunk does not exceed max_chars unless a single section is larger.
    """
    import re
    if len(text) <= max_chars:
        return [text]

    # Pattern to detect headings, clauses, sections (e.g. Section 1, Clause A, ARTICLE II, 1.1, etc.)
    heading_pattern = re.compile(
        r'^(?:SECTION|CLAUSE|ARTICLE|SCHEDULE|EXHIBIT|\d+\.\d+|\d+\.)\b', 
        re.IGNORECASE
    )

    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        stripped = line.strip()
        # If line starts a new section and we are already holding content
        if heading_pattern.match(stripped) and current_chunk:
            # Check if adding this section would exceed limit
            if current_length >= max_chars:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        current_chunk.append(line)
        current_length += len(line) + 1

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    # If any single chunk is still larger than max_chars, split it by paragraph/length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Split by paragraph
            paragraphs = chunk.split('\n\n')
            sub_chunk = []
            sub_length = 0
            for para in paragraphs:
                if sub_length + len(para) > max_chars and sub_chunk:
                    final_chunks.append('\n\n'.join(sub_chunk))
                    sub_chunk = []
                    sub_length = 0
                sub_chunk.append(para)
                sub_length += len(para) + 2
            if sub_chunk:
                final_chunks.append('\n\n'.join(sub_chunk))

    return final_chunks

def generate_pdf_report(final_state: dict) -> bytes:
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151'),
        spaceAfter=10
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyTextCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Header Title
    story.append(Paragraph("LegalValidate AI - Document Audit Report", title_style))
    story.append(Paragraph("Automated Legal Validation, Risk Detection & Verification Trail", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Metadata Box
    is_legal = final_state.get('is_legal', False)
    status_text = "APPROVED LEGAL DOCUMENT" if is_legal else "NON-LEGAL / NEEDS MANUAL REVIEW"
    status_color = '#10B981' if is_legal else '#EF4444'
    
    metadata_data = [
        [Paragraph("<b>Document Type:</b>", body_style), Paragraph(final_state.get('document_type', 'Unknown'), body_style)],
        [Paragraph("<b>Validation Status:</b>", body_style), Paragraph(f"<font color='{status_color}'><b>{status_text}</b></font>", body_style)],
        [Paragraph("<b>Classification Reason:</b>", body_style), Paragraph(final_state.get('classification_reason', 'N/A'), body_style)]
    ]
    
    t = Table(metadata_data, colWidths=[130, 370])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor('#E5E7EB')),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Plain Language Summary & Explanation
    story.append(Paragraph("Simplified Document Summary", h1_style))
    story.append(Paragraph(final_state.get('summary', 'No summary generated.'), body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Plain Language Explanation", h1_style))
    story.append(Paragraph(final_state.get('simplified_explanation', 'No explanation generated.'), body_style))
    story.append(Spacer(1, 20))
    
    # Flagged Risks & Verdicts
    story.append(Paragraph("Identified Risks & Critic Verification", h1_style))
    risks = final_state.get('risks', [])
    if not risks:
        story.append(Paragraph("No critical legal risks or non-compliance issues identified.", body_style))
    else:
        critic_lookup = {r.get("risk", "").strip().lower(): r for r in final_state.get("critic_reviews", [])}
        
        # Table of risks
        risk_table_data = [[
            Paragraph("<b>Detected Risk</b>", bold_body_style),
            Paragraph("<b>Critic Verdict</b>", bold_body_style),
            Paragraph("<b>Verification Reason / Notes</b>", bold_body_style)
        ]]
        
        for risk in risks:
            review = critic_lookup.get(risk.strip().lower())
            verdict = "N/A"
            reason = "No verification performed."
            verdict_color = '#374151'
            
            if review:
                verdict = review.get("verdict", "Unknown")
                reason = review.get("reason", "")
                if verdict == "Confirmed":
                    verdict_color = '#DC2626' # Red
                elif verdict == "Downgraded":
                    verdict_color = '#D97706' # Orange
                else:
                    verdict_color = '#4B5563' # Grey
            
            risk_table_data.append([
                Paragraph(risk, body_style),
                Paragraph(f"<font color='{verdict_color}'><b>{verdict}</b></font>", body_style),
                Paragraph(reason, body_style)
            ])
            
        rt = Table(risk_table_data, colWidths=[150, 90, 260])
        rt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E7EB')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ]))
        story.append(rt)
        
    # Build document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
