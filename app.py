import streamlit as st
import os
from dotenv import load_dotenv
from utils import extract_text_from_pdf, clean_text, generate_pdf_report
from graph import create_graph, comparison_app
from db import (
    save_analysis,
    save_comparison,
    get_all_analyses,
    get_analysis_by_id,
    get_all_comparisons,
    get_comparison_by_id
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LegalValidate AI | Document Review",
    page_icon="🛡️",
    layout="wide"
)

# LegalValidate.ai Inspired CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E40AF !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    /* Header Section */
    .header-section {
        padding: 1.5rem 0;
        background: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        text-align: left;
        margin-bottom: 2rem;
    }
    .header-section h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    
    /* Dashboard Metrics Bar */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        flex: 1;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.25rem;
        color: #111827;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    
    /* Insight Cards */
    .insight-card {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .insight-card-risk {
        border-top: 4px solid #EF4444;
    }
    .insight-card-success {
        border-top: 4px solid #10B981;
    }
    .insight-card-info {
        border-top: 4px solid #3B82F6;
    }
    
    /* Sidebar Fixes */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #374151;
    }
    [data-testid="stSidebar"] * {
        color: #F9FAFB !important;
    }
    
    /* Risk Highlighting */
    .risk-item {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
        color: #991B1B;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1E40AF !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1E3A8A !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def render_single_analysis_dashboard(final_state):
    """Renders the dashboard metrics and insight cards for a single document review."""
    is_legal = final_state.get('is_legal')
    risks = final_state.get('risks', [])
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Document Type</div>
            <div class="metric-value">{final_state.get('document_type', 'Unknown')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Legal Status</div>
            <div class="metric-value" style="color: {'#10B981' if is_legal else '#EF4444'}">{'APPROVED LEGAL' if is_legal else 'NON-LEGAL / REVIEW'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Detected Risks</div>
            <div class="metric-value" style="color: {'#111827' if not risks else '#EF4444'}">{len(risks)} found</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Primary Insights Row
    row1_c1, row1_c2 = st.columns(2)
    
    with row1_c1:
        st.markdown('<div class="insight-card insight-card-risk">', unsafe_allow_html=True)
        st.header("⚠️ Risk Intelligence & Verification")
        if risks:
            critic_lookup = {r.get("risk", "").strip().lower(): r for r in final_state.get("critic_reviews", [])}
            for risk in risks:
                review = critic_lookup.get(risk.strip().lower())
                if review:
                    verdict = review.get("verdict", "Unknown")
                    reason = review.get("reason", "")
                    
                    if verdict == "Confirmed":
                        color = "#991B1B"
                        bg = "#FEF2F2"
                        border = "#EF4444"
                    elif verdict == "Downgraded":
                        color = "#92400E"
                        bg = "#FEF3C7"
                        border = "#F59E0B"
                    else: # Removed
                        color = "#374151"
                        bg = "#F3F4F6"
                        border = "#9CA3AF"
                    
                    st.markdown(f"""
                    <div style="background: {bg}; border-left: 4px solid {border}; padding: 1rem; margin-bottom: 1rem; border-radius: 4px; color: {color};">
                        <strong>⚠️ {risk}</strong><br/>
                        <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Verdict: {verdict}</span><br/>
                        <span style="font-size: 0.85rem; opacity: 0.9;">Reason: {reason}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-item">⚠️ {risk}</div>', unsafe_allow_html=True)
        else:
            st.info("No critical legal risks or non-compliance issues identified.")
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_c2:
        st.markdown('<div class="insight-card insight-card-info">', unsafe_allow_html=True)
        st.header("💡 Simplified Review")
        st.markdown(f"**Quick Summary:**\n{final_state.get('summary')}")
        st.markdown("---")
        st.markdown(f"**Plain Language Explanation:**\n{final_state.get('simplified_explanation')}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Secondary Details Row
    st.markdown("---")
    row2_c1, row2_c2 = st.columns([1, 1])
    
    with row2_c1:
        with st.expander("📄 Document Breakdown & Key Clauses", expanded=True):
            st.write(f"**Classification Note:** {final_state.get('classification_reason')}")
            if not is_legal and final_state.get('problematic_explanation'):
                st.warning(final_state.get('problematic_explanation'))
            
            st.markdown("#### Key Sections Identified:")
            for clause in final_state.get('key_clauses', []):
                st.markdown(f"- {clause}")

    with row2_c2:
        with st.expander("📊 Raw Intelligence Data"):
            st.json({
                "document_type": final_state.get('document_type'),
                "is_legal": final_state.get('is_legal'),
                "summary": final_state.get('summary'),
                "key_clauses": final_state.get('key_clauses'),
                "risks": final_state.get('risks'),
                "critic_reviews": final_state.get('critic_reviews'),
                "simplified_explanation": final_state.get('simplified_explanation')
            })

def render_comparison_dashboard(comp_results):
    """Renders semantic redline and comparison clause cards."""
    st.markdown("---")
    st.markdown("### 🔍 Semantic Redline & Comparison Results")
    st.markdown("The AI has semantically aligned the clauses of the original and modified documents. Below are the detected differences and risk assessments:")
    
    comparisons = comp_results.get("comparisons", [])
    if not comparisons:
        st.info("No clause differences or modifications detected between the two documents.")
    else:
        for idx, item in enumerate(comparisons):
            direction = item.get("risk_direction", "No Change")
            if direction == "Increase":
                badge_color = "#991B1B"
                badge_bg = "#FEF2F2"
                badge_border = "#EF4444"
                badge_text = "🔴 Risk Increased"
            elif direction == "Decrease":
                badge_color = "#065F46"
                badge_bg = "#ECFDF5"
                badge_border = "#10B981"
                badge_text = "🟢 Risk Decreased"
            else:
                badge_color = "#374151"
                badge_bg = "#F3F4F6"
                badge_border = "#9CA3AF"
                badge_text = "⚪ No Change"
                
            st.markdown(f"""
            <div style="border: 1px solid #E5E7EB; border-radius: 8px; margin-bottom: 2rem; background: white; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="background: {badge_bg}; border-bottom: 1px solid {badge_border}; padding: 0.75rem 1rem; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #1F2937;">Clause Pair #{idx+1}</span>
                    <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {badge_color}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">
                        {badge_text}
                    </span>
                </div>
                <div style="padding: 1.25rem;">
                    <div style="display: flex; gap: 1.5rem; margin-bottom: 1rem;">
                        <div style="flex: 1; min-width: 0; background: #F9FAFB; padding: 1rem; border-radius: 6px; border-left: 3px solid #D1D5DB;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #6B7280; margin-bottom: 0.5rem;">Original Clause</div>
                            <div style="font-size: 0.9rem; color: #374151; white-space: pre-wrap;">{item.get('original_clause')}</div>
                        </div>
                        <div style="flex: 1; min-width: 0; background: #FFFBEB; padding: 1rem; border-radius: 6px; border-left: 3px solid #FCD34D;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #B45309; margin-bottom: 0.5rem;">Modified Clause</div>
                            <div style="font-size: 0.9rem; color: #78350F; white-space: pre-wrap;">{item.get('modified_clause')}</div>
                        </div>
                    </div>
                    <div style="border-top: 1px solid #F3F4F6; padding-top: 1rem; margin-top: 1rem;">
                        <div style="margin-bottom: 0.5rem;"><strong>📝 Modification:</strong> {item.get('change_description')}</div>
                        <div style="color: {badge_color}; font-weight: 600;"><strong>💡 Risk Verdict Reason:</strong> {item.get('reason')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ LegalValidate Engine")
    
    # Check for API Key in environment
    env_api_key = os.getenv("GROQ_API_KEY")
    if not env_api_key or env_api_key == "your_groq_api_key_here":
        st.error("⚠️ GROQ_API_KEY missing.")
        st.info("Please set the key in your `.env` file to activate the AI.")
        api_key = None
    else:
        st.success("✅ AI Engine Online")
        api_key = env_api_key
        
    st.markdown("---")
    st.markdown("### 🛠️ Mode")
    app_mode = st.selectbox(
        "Select Operation Mode",
        ["Single Document Review", "Double Document Comparison", "History"],
        key="app_mode"
    )
    
    if app_mode == "Single Document Review" and st.session_state.get("graph_status") == "completed" and st.session_state.get("final_state"):
        st.markdown("---")
        st.markdown("### 📊 Export Report")
        try:
            pdf_bytes = generate_pdf_report(st.session_state.final_state)
            st.download_button(
                label="📥 Download PDF Audit Report",
                data=pdf_bytes,
                file_name=f"LegalValidate_Report_{st.session_state.final_state.get('document_type', 'Document').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as pdf_err:
            st.error(f"Failed to generate PDF: {pdf_err}")
            
    st.markdown("---")
    st.markdown("#### About")
    st.caption("LegalValidate AI provides instant, simplified document review and automated risk analysis for non-experts.")

# Custom Header
st.markdown("""
<div class="header-section">
    <h1>🛡️ LegalValidate AI</h1>
    <p style="color: #6B7280; font-size: 1.1rem;">Automated Contract Review & Document Insight</p>
</div>
""", unsafe_allow_html=True)

# Check for mode change to reset state
if "prev_app_mode" not in st.session_state:
    st.session_state.prev_app_mode = app_mode

if st.session_state.prev_app_mode != app_mode:
    st.session_state.prev_app_mode = app_mode
    st.session_state.graph_status = 'idle'
    st.session_state.final_state = None
    st.session_state.edited_clauses = None
    st.session_state.cleaned_text = None
    st.session_state.comp_status = 'idle'
    st.session_state.comp_results = None
    st.session_state.cleaned_text_a = None
    st.session_state.cleaned_text_b = None

# Main Flow branching
if app_mode == "Single Document Review":
    col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
    with col_up2:
        uploaded_file = st.file_uploader("📂 Upload Legal Document for Review", type=["pdf", "txt"], key="single_uploader")

    if uploaded_file and api_key:
        file_bytes = uploaded_file.read()
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Check if file changed, if so, reset state
        uploaded_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_id") != uploaded_file_id:
            import uuid
            st.session_state.current_file_id = uploaded_file_id
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.graph_status = 'idle'
            st.session_state.final_state = None
            st.session_state.edited_clauses = None
            st.session_state.cleaned_text = None

        # Sidebar reset control
        if st.sidebar.button("🔄 Reset Analysis", use_container_width=True):
            st.session_state.graph_status = 'idle'
            st.session_state.final_state = None
            st.session_state.edited_clauses = None
            st.session_state.cleaned_text = None
            st.rerun()

        # Step 1: Idle - Show the trigger button
        if st.session_state.graph_status == 'idle':
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🚀 Analyze Now", use_container_width=True):
                    with st.spinner("🚀 Extracting and preparing text..."):
                        if file_extension == 'pdf':
                            raw_text = extract_text_from_pdf(file_bytes)
                        else:
                            raw_text = file_bytes.decode("utf-8")
                        st.session_state.cleaned_text = clean_text(raw_text)
                        st.session_state.graph_status = 'processing_initial'
                        st.rerun()

        # Handle the running state machine
        if st.session_state.graph_status != 'idle':
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            app = create_graph()
            
            if st.session_state.graph_status == 'processing_initial':
                status_box = st.status("🧠 Processing initial analysis...", expanded=True)
                status_box.write("🔍 **Analyzer & Classifier Agents**: Scanning structure and assessing legal standing...")
                try:
                    initial_state = {
                        "text": st.session_state.cleaned_text,
                        "api_key": api_key,
                        "document_type": None,
                        "is_legal": None,
                        "key_clauses": [],
                        "risks": [],
                        "summary": None,
                        "simplified_explanation": None,
                        "classification_reason": None,
                        "problematic_explanation": None,
                        "critic_reviews": []
                    }
                    
                    final_state = initial_state.copy()
                    for output in app.stream(initial_state, config):
                        for node_name, node_state in output.items():
                            if node_name == "analyzer":
                                status_box.write("✅ **Analyzer Agent**: Finished document breakdown.")
                            elif node_name == "classifier":
                                status_box.write("✅ **Validator Agent**: Legal standing assessed.")
                            final_state.update(node_state)
                            
                    st.session_state.final_state = final_state
                    st.session_state.edited_clauses = final_state.get("key_clauses", [])
                    st.session_state.graph_status = 'paused'
                    status_box.update(label="⏸️ Paused for Human-in-the-Loop Review", state="complete", expanded=True)
                    st.rerun()
                except Exception as e:
                    st.session_state.graph_status = 'idle'
                    status_box.update(label="❌ Analysis Failed", state="error", expanded=True)
                    st.error(f"Engine Error: {str(e)}")
                    st.stop()
                    
            elif st.session_state.graph_status == 'paused':
                st.markdown('<div class="insight-card insight-card-info" style="margin-bottom: 2rem;">', unsafe_allow_html=True)
                st.markdown("### 🧑‍⚖️ Human-in-the-Loop: Review & Approve Key Clauses")
                st.info("The Analyzer agent has extracted key clauses from the document. Please review and edit them below before resuming the risk detection and summarization.")
                
                # Simple text area for line-by-line editing
                clauses_text = "\n".join(st.session_state.edited_clauses)
                edited_text = st.text_area("Extracted Key Clauses (one per line):", value=clauses_text, height=250)
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🚀 Approve & Resume", use_container_width=True):
                        updated_clauses = [line.strip() for line in edited_text.split("\n") if line.strip()]
                        st.session_state.edited_clauses = updated_clauses
                        st.session_state.graph_status = 'processing_resume'
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()
                
            elif st.session_state.graph_status == 'processing_resume':
                status_box = st.status("🧠 Resuming analysis...", expanded=True)
                status_box.write("⚡ **Updating State with Approved Clauses...**")
                try:
                    # Update the state of key_clauses in the checkpointer
                    app.update_state(config, {"key_clauses": st.session_state.edited_clauses}, as_node="analyzer")
                    
                    status_box.write("🏃 **Running downstream agents (Risk, Critic, Explainer)...**")
                    
                    for output in app.stream(None, config):
                        for node_name, node_state in output.items():
                            if node_name == "risk_detector":
                                status_box.write("✅ **Risk Agent**: Risk assessment complete.")
                            elif node_name == "critic":
                                status_box.write("✅ **Critic Agent**: Risk verification complete.")
                            elif node_name == "explainer":
                                status_box.write("✅ **Summarizer Agent**: Plain language translation ready.")
                    
                    # Fetch full merged state from the checkpointer
                    current_state = app.get_state(config)
                    st.session_state.final_state = dict(current_state.values)
                    st.session_state.graph_status = 'completed'

                    # Save result to SQLite database
                    try:
                        fname = uploaded_file.name if uploaded_file else "document.pdf"
                        save_analysis(fname, st.session_state.final_state)
                    except Exception as db_err:
                        st.warning(f"Failed to persist analysis to DB: {db_err}")

                    status_box.update(label="✅ Analysis Complete", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    st.session_state.graph_status = 'paused'
                    status_box.update(label="❌ Resume Failed", state="error", expanded=True)
                    st.error(f"Engine Error: {str(e)}")
                    st.stop()
                    
            elif st.session_state.graph_status == 'completed':
                render_single_analysis_dashboard(st.session_state.final_state)

    elif not api_key:
        st.warning("👈 Please configure your Groq API Key in the `.env` file to begin.")
    else:
        st.info("📂 Waiting for file upload to start automated review.")

elif app_mode == "Double Document Comparison":
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_file_a = st.file_uploader("📂 Upload Original Document", type=["pdf", "txt"], key="compare_uploader_a")
    with col_up2:
        uploaded_file_b = st.file_uploader("📂 Upload Modified / Redlined Document", type=["pdf", "txt"], key="compare_uploader_b")

    if uploaded_file_a and uploaded_file_b and api_key:
        file_bytes_a = uploaded_file_a.read()
        file_bytes_b = uploaded_file_b.read()
        file_extension_a = uploaded_file_a.name.split('.')[-1].lower()
        file_extension_b = uploaded_file_b.name.split('.')[-1].lower()
        
        # Check if files changed, if so, reset comparison state
        comp_file_id = f"{uploaded_file_a.name}_{uploaded_file_a.size}_{uploaded_file_b.name}_{uploaded_file_b.size}"
        if st.session_state.get("current_comp_file_id") != comp_file_id:
            import uuid
            st.session_state.current_comp_file_id = comp_file_id
            st.session_state.comp_thread_id = str(uuid.uuid4())
            st.session_state.comp_status = 'idle'
            st.session_state.comp_results = None
            st.session_state.cleaned_text_a = None
            st.session_state.cleaned_text_b = None

        # Sidebar reset control
        if st.sidebar.button("🔄 Reset Comparison", use_container_width=True):
            st.session_state.comp_status = 'idle'
            st.session_state.comp_results = None
            st.session_state.cleaned_text_a = None
            st.session_state.cleaned_text_b = None
            st.rerun()

        # Step 1: Idle - Show comparison trigger button
        if st.session_state.comp_status == 'idle':
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🚀 Compare Documents", use_container_width=True):
                    with st.spinner("🚀 Extracting and preparing documents..."):
                        if file_extension_a == 'pdf':
                            raw_text_a = extract_text_from_pdf(file_bytes_a)
                        else:
                            raw_text_a = file_bytes_a.decode("utf-8")
                            
                        if file_extension_b == 'pdf':
                            raw_text_b = extract_text_from_pdf(file_bytes_b)
                        else:
                            raw_text_b = file_bytes_b.decode("utf-8")
                            
                        st.session_state.cleaned_text_a = clean_text(raw_text_a)
                        st.session_state.cleaned_text_b = clean_text(raw_text_b)
                        st.session_state.comp_status = 'processing'
                        st.rerun()

        # Handle running comparison
        if st.session_state.comp_status != 'idle':
            comp_config = {"configurable": {"thread_id": st.session_state.comp_thread_id}}
            
            if st.session_state.comp_status == 'processing':
                status_box = st.status("🧠 Semantically aligning and comparing documents...", expanded=True)
                status_box.write("🔍 **Comparison Agent**: Running clause alignment and calculating semantic similarities...")
                try:
                    initial_state = {
                        "text_a": st.session_state.cleaned_text_a,
                        "text_b": st.session_state.cleaned_text_b,
                        "api_key": api_key,
                        "comparisons": []
                    }
                    
                    final_comp_state = initial_state.copy()
                    for output in comparison_app.stream(initial_state, comp_config):
                        for node_name, node_state in output.items():
                            if node_name == "compare":
                                status_box.write("✅ **Comparison Agent**: Clause alignment and risk analysis complete.")
                            final_comp_state.update(node_state)
                            
                    st.session_state.comp_results = final_comp_state
                    st.session_state.comp_status = 'completed'

                    # Save comparison to SQLite database
                    try:
                        fname_a = uploaded_file_a.name if uploaded_file_a else "original.pdf"
                        fname_b = uploaded_file_b.name if uploaded_file_b else "modified.pdf"
                        save_comparison(fname_a, fname_b, final_comp_state)
                    except Exception as db_err:
                        st.warning(f"Failed to persist comparison to DB: {db_err}")

                    status_box.update(label="✅ Comparison Complete", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    st.session_state.comp_status = 'idle'
                    status_box.update(label="❌ Comparison Failed", state="error", expanded=True)
                    st.error(f"Engine Error: {str(e)}")
                    st.stop()
                    
            elif st.session_state.comp_status == 'completed':
                render_comparison_dashboard(st.session_state.comp_results)

    elif not api_key:
        st.warning("👈 Please configure your Groq API Key in the `.env` file to begin.")
    else:
        st.info("📂 Waiting for both the Original and Modified files to start comparison review.")

elif app_mode == "History":
    st.markdown("## 📜 Analysis & Comparison History")
    st.caption("View stored document reviews and semantic comparisons without re-running LLMs.")
    
    tab_analyses, tab_comparisons = st.tabs(["📄 Single Document Reviews", "🔍 Dual Document Comparisons"])
    
    with tab_analyses:
        analyses = get_all_analyses()
        if not analyses:
            st.info("No past document reviews found in database.")
        else:
            options = {
                f"#{item['id']} | {item['filename']} ({item['timestamp']}) - {item['document_type']} [{item['classification']}]": item['id']
                for item in analyses
            }
            selected_label = st.selectbox("Select a past document analysis to inspect:", list(options.keys()))
            if selected_label:
                selected_id = options[selected_label]
                record = get_analysis_by_id(selected_id)
                if record:
                    st.markdown("---")
                    st.markdown(f"### 📂 Stored Analysis: {record['filename']}")
                    st.caption(f"**Saved Date:** {record['timestamp']} | **Database ID:** #{record['id']}")
                    
                    try:
                        pdf_bytes = generate_pdf_report(record["full_result"])
                        st.download_button(
                            label="📥 Download Stored PDF Audit Report",
                            data=pdf_bytes,
                            file_name=f"LegalValidate_Report_DB{record['id']}_{record['filename'].replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as pdf_err:
                        st.warning(f"Could not generate PDF for history item: {pdf_err}")
                        
                    st.markdown("---")
                    render_single_analysis_dashboard(record["full_result"])
                    
    with tab_comparisons:
        comparisons_list = get_all_comparisons()
        if not comparisons_list:
            st.info("No past document comparisons found in database.")
        else:
            comp_options = {
                f"#{item['id']} | {item['filename_a']} vs {item['filename_b']} ({item['timestamp']}) - {item['comparison_count']} changes": item['id']
                for item in comparisons_list
            }
            selected_comp_label = st.selectbox("Select a past document comparison to inspect:", list(comp_options.keys()))
            if selected_comp_label:
                selected_comp_id = comp_options[selected_comp_label]
                comp_record = get_comparison_by_id(selected_comp_id)
                if comp_record:
                    st.markdown("---")
                    st.markdown(f"### 📂 Stored Comparison: {comp_record['filename_a']} vs {comp_record['filename_b']}")
                    st.caption(f"**Saved Date:** {comp_record['timestamp']} | **Database ID:** #{comp_record['id']}")
                    st.markdown("---")
                    render_comparison_dashboard(comp_record["full_result"])
