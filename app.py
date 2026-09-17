import streamlit as st
import os
import uuid
import json
import datetime
from dotenv import load_dotenv

from utils import extract_document_text, clean_text, generate_pdf_report
from graph import create_graph, comparison_app
from db import (
    save_analysis,
    update_analysis_human_decisions,
    save_comparison,
    get_all_analyses,
    get_analysis_by_id,
    get_all_comparisons,
    get_comparison_by_id
)

# Load environment variables
load_dotenv()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="LegalValidate AI | Multi-Agent Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# SAMPLE CONTRACT TEST CORPUS FOR 1-CLICK DEMOS
# =====================================================================
SAMPLE_DOCUMENTS = {
    "--- Select a Sample Contract ---": None,
    "⚠️ Risky Commercial Lease (Deposit Forfeiture & Unannounced Entry)": """COMMERCIAL REAL ESTATE LEASE AGREEMENT
This Commercial Real Estate Lease Agreement is entered into on September 15, 2026, by and between Metro Plaza Holdings LLC ("Landlord") and Apex Retail Inc ("Tenant").
1. Premises: Suite 400, Metro Tower, 100 Main St, Dover, DE.
2. Term: The initial term shall be for five (5) years commencing October 1, 2026.
3. Security Deposit & Forfeiture: Tenant shall deposit $10,000 as a security deposit. Landlord shall retain the entire security deposit unconditionally upon lease expiration or termination, regardless of the physical condition of the premises or presence of damage.
4. Landlord Inspection & Entry: Landlord and its agents reserve the unrestricted right to enter the leased premises at any hour of the day or night without prior notice to Tenant for inspection, maintenance, or any other purpose.
5. Rent Escalation: Monthly base rent shall increase by 15% compounded annually on each anniversary of the Effective Date.
6. Relocation: Landlord reserves the right, at its sole discretion, to relocate Tenant to a smaller suite within the building upon providing five (5) days advance written notice.""",

    "✅ Standard Mutual NDA (Balanced & Customary)": """MUTUAL NON-DISCLOSURE AGREEMENT
This Mutual Non-Disclosure Agreement is made and entered into by and between Alpha Technologies Corp ("Alpha") and Beta Solutions Inc ("Beta").
1. Purpose: The parties wish to explore a potential strategic business collaboration.
2. Definition of Confidential Information: "Confidential Information" means all non-public proprietary technical and business information disclosed by one party to the other, marked as confidential or that reasonably should be understood to be confidential.
3. Exclusions: Confidential Information shall not include information that: (a) is or becomes publicly known through no breach; (b) was already known to the receiving party; (c) is independently developed without reference to the disclosing party's confidential information.
4. Term: The confidentiality obligations shall survive for a period of three (3) years from the date of initial disclosure.
5. Return of Materials: Upon written request, each party shall return or destroy all confidential materials, provided that one archival copy may be retained solely for legal and regulatory compliance.
6. Governing Law: This Agreement shall be governed by and construed under the laws of the State of Delaware.""",

    "🚨 High-Risk Executive Employment (Global Non-Compete & IP Forfeiture)": """EXECUTIVE EMPLOYMENT AGREEMENT
This Executive Employment Agreement is entered into between FutureTech Enterprise LLC ("Company") and Jane Smith ("Executive").
1. Position: Chief Technology Officer. Base salary: $250,000 per annum.
2. Restrictive Covenant & Non-Compete: Executive agrees that during employment and for a period of five (5) years following termination for any reason, Executive shall not directly or indirectly engage in, advise, or invest in any technology enterprise anywhere globally.
3. Intellectual Property Assignment: All inventions, code, software, concepts, and designs created by Executive during the employment term, whether developed during working hours or on personal time, with or without Company resources, shall be the sole and exclusive property of Company.
4. Termination & Severance: Company may terminate Executive's employment at any time without cause, with zero advance notice, and with zero severance compensation. Executive waives all claims to accrued but unpaid bonuses upon termination.""",

    "🚫 Non-Legal Document (Spaghetti Carbonara Recipe)": """CLASSIC ITALIAN SPAGHETTI CARBONARA
Ingredients:
- 400g spaghetti pasta
- 150g cured guanciale or pancetta, cubed
- 4 large fresh egg yolks + 1 whole egg
- 80g finely grated Pecorino Romano cheese
- Freshly ground black pepper
- Salt for pasta water

Cooking Instructions:
1. Bring a large pot of salted water to a rolling boil. Add spaghetti and cook until al dente.
2. In a skillet over medium heat, fry the guanciale until golden and crispy. Remove from heat and reserve the rendered fat.
3. In a bowl, whisk egg yolks, whole egg, grated Pecorino, and abundant black pepper into a thick cream.
4. Drain pasta, reserving 1/2 cup of starchy pasta cooking water.
5. Toss pasta with guanciale and rendered fat, then remove pan from direct flame.
6. Pour in egg and cheese mixture, stirring vigorously and adding pasta water spoonful by spoonful until a rich, silky sauce forms. Serve immediately with extra Pecorino."""
}


# =====================================================================
# PREMIUM UI STYLES (THEME-RESILIENT, GLASSMORPHISM & CRISP CARDS)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #1E40AF 100%);
        color: #FFFFFF !important;
        padding: 1.75rem 2.25rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-banner h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .header-banner p {
        color: #93C5FD !important;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Disclaimer */
    .disclaimer-banner {
        background: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        color: #78350F !important;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    /* KPI Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 1.1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.72rem;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }
    .metric-val {
        font-size: 1.35rem;
        color: #0F172A !important;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    
    /* Risk Cards */
    .risk-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    .risk-critical { border-left: 6px solid #DC2626; }
    .risk-high { border-left: 6px solid #EA580C; }
    .risk-medium { border-left: 6px solid #F59E0B; }
    .risk-low { border-left: 6px solid #2563EB; }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-critical { background: #FEE2E2; color: #991B1B !important; border: 1px solid #F87171; }
    .badge-high { background: #FFEDD5; color: #9A3412 !important; border: 1px solid #FB923C; }
    .badge-medium { background: #FEF3C7; color: #92400E !important; border: 1px solid #FCD34D; }
    .badge-low { background: #DBEAFE; color: #1E40AF !important; border: 1px solid #93C5FD; }
    .badge-valid { background: #D1FAE5; color: #065F46 !important; border: 1px solid #6EE7B7; }
    .badge-removed { background: #F1F5F9; color: #475569 !important; border: 1px solid #CBD5E1; }
    
    /* Code Excerpt Box */
    .clause-box {
        background: #F8FAFC;
        border-left: 3px solid #64748B;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #1E293B !important;
        line-height: 1.45;
    }
    
    /* Form & Action Box */
    .hitl-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }

    /* Redline Box */
    .diff-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .diff-a {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        color: #7F1D1D !important;
        font-size: 0.88rem;
    }
    .diff-b {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        color: #14532D !important;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# UI RENDERING FUNCTIONS
# =====================================================================

def render_kpi_bar(final_state: dict):
    is_legal = final_state.get('is_legal', False)
    risks = final_state.get('risks', []) or []
    critic_reviews = final_state.get('critic_reviews', []) or []
    critic_map = {r.get('risk_id'): r for r in critic_reviews if isinstance(r, dict)}
    
    crit_count = sum(1 for r in risks if isinstance(r, dict) and critic_map.get(r.get('risk_id'), {}).get('verified_severity', r.get('severity')) == 'CRITICAL')
    high_count = sum(1 for r in risks if isinstance(r, dict) and critic_map.get(r.get('risk_id'), {}).get('verified_severity', r.get('severity')) == 'HIGH')
    med_count = sum(1 for r in risks if isinstance(r, dict) and critic_map.get(r.get('risk_id'), {}).get('verified_severity', r.get('severity')) == 'MEDIUM')
    low_count = sum(1 for r in risks if isinstance(r, dict) and critic_map.get(r.get('risk_id'), {}).get('verified_severity', r.get('severity')) == 'LOW')
    
    doc_type = final_state.get('document_type', 'Unknown')
    legal_badge = "✅ VALID CONTRACT" if is_legal else "🚫 NON-LEGAL"
    legal_color = "#059669" if is_legal else "#DC2626"
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Classification</div>
            <div class="metric-val" style="font-size: 1.1rem;">{doc_type}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Legal Standing</div>
            <div class="metric-val" style="color: {legal_color}; font-size: 1.1rem;">{legal_badge}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Critical Hazards</div>
            <div class="metric-val" style="color: #DC2626;">{crit_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">High Risks</div>
            <div class="metric-val" style="color: #EA580C;">{high_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Med / Low Risks</div>
            <div class="metric-val" style="color: #D97706;">{med_count + low_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_single_analysis_dashboard(final_state: dict):
    render_kpi_bar(final_state)
    
    tabs = st.tabs([
        "📊 Executive Overview",
        "🛡️ Risk Intelligence",
        "🔍 Clause Explorer & RAG Trail",
        "💡 Plain-Language Summary",
        "🧑‍⚖️ Human Audit Decisions",
        "📄 Raw Intelligence JSON"
    ])
    
    # TAB 1: EXECUTIVE OVERVIEW
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### 📋 Executive Summary")
            summary = final_state.get("summary", "No summary generated.")
            st.info(summary)
            
            st.markdown("### 💡 Key Takeaways")
            simplified = final_state.get("simplified_explanation", "No explanation generated.")
            st.write(simplified)
            
        with col2:
            st.markdown("### 📑 Contract Details")
            analysis = final_state.get("document_analysis", {}) or {}
            if analysis:
                st.markdown(f"**Parties:** {', '.join(analysis.get('parties', [])) or 'Not explicitly stated'}")
                st.markdown(f"**Effective Date:** {analysis.get('effective_date') or 'Unspecified'}")
                st.markdown(f"**Term / Duration:** {analysis.get('term') or 'Unspecified'}")
                st.markdown(f"**Renewal:** {analysis.get('renewal') or 'Unspecified'}")
                st.markdown(f"**Payment Terms:** {analysis.get('payment_terms') or 'Unspecified'}")
                st.markdown(f"**Governing Law:** {analysis.get('governing_law_and_jurisdiction') or 'Unspecified'}")
                
                missing = analysis.get("missing_sections", [])
                if missing:
                    st.markdown("#### 🚨 Missing Standard Safeguards:")
                    for m in missing:
                        st.markdown(f"- ⚠️ **{m}**")
            else:
                st.caption("Detailed structural metadata not available.")

    # TAB 2: RISK INTELLIGENCE
    with tabs[1]:
        risks = final_state.get("risks", []) or []
        critic_reviews = final_state.get("critic_reviews", []) or []
        critic_map = {r.get("risk_id"): r for r in critic_reviews if isinstance(r, dict)}
        
        if not risks:
            st.success("✅ No critical legal risks or non-compliance vulnerabilities detected.")
        else:
            col_f1, col_f2 = st.columns([2, 2])
            with col_f1:
                sev_filter = st.multiselect(
                    "Filter Severity:",
                    ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                )
            with col_f2:
                search_kw = st.text_input("Search Risks by Keyword:", placeholder="e.g. deposit, notice, indemnity")
                
            for idx, r in enumerate(risks, start=1):
                if not isinstance(r, dict):
                    continue
                r_id = r.get("risk_id", f"risk_{idx}")
                c_review = critic_map.get(r_id, {})
                c_sev = c_review.get("verified_severity", r.get("severity", "MEDIUM"))
                c_valid = c_review.get("is_valid", True)
                c_reason = c_review.get("reason", "Audited by critic agent.")
                
                if c_sev not in sev_filter:
                    continue
                    
                if search_kw:
                    full_text = f"{r.get('risk_type')} {r.get('clause')} {r.get('explanation')}".lower()
                    if search_kw.lower() not in full_text:
                        continue
                        
                sev_class = f"risk-{c_sev.lower()}"
                badge_class = f"badge-{c_sev.lower()}"
                
                st.markdown(f"""
                <div class="risk-card {sev_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: #0F172A;">#{idx}. {r.get('risk_type', 'Contract Hazard')}</span>
                        <div>
                            <span class="badge {badge_class}">{c_sev}</span>
                            <span class="badge {'badge-valid' if c_valid else 'badge-removed'}">Critic: {'Verified' if c_valid else 'Downgraded/Removed'}</span>
                        </div>
                    </div>
                    <div class="clause-box">
                        <strong>Target Clause:</strong> "{r.get('clause', '')}"
                    </div>
                    <div style="font-size: 0.92rem; color: #1E293B; margin-bottom: 0.5rem;">
                        <strong>Risk Analysis:</strong> {r.get('explanation', '')}
                    </div>
                    <div style="font-size: 0.92rem; color: #059669; margin-bottom: 0.5rem;">
                        <strong>💡 Recommendation:</strong> {r.get('recommendation', '')}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; border-top: 1px solid #F1F5F9; padding-top: 0.5rem; margin-top: 0.5rem;">
                        <strong>Critic Verification Note:</strong> {c_reason}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # TAB 3: CLAUSE EXPLORER & RAG TRAIL
    with tabs[2]:
        st.markdown("### 📚 Grounded RAG Retrieval & Baseline Standards")
        st.caption("Inspect how each contract clause was evaluated against the standard legal baseline knowledge corpus.")
        
        risks = final_state.get("risks", []) or []
        if not risks:
            st.info("No clause anomalies identified.")
        else:
            for idx, r in enumerate(risks, start=1):
                if not isinstance(r, dict):
                    continue
                with st.expander(f"Clause #{idx}: {r.get('risk_type')} [{r.get('severity')}]", expanded=(idx==1)):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Contract Document Excerpt:**")
                        st.code(r.get('clause', ''), language="text")
                    with col_b:
                        st.markdown("**Retrieved Baseline Reference Standard:**")
                        sources = r.get('source_reference', []) or []
                        if sources and sources[0] != "Insufficient retrieved evidence":
                            for s in sources:
                                st.info(s)
                        else:
                            st.warning("No standard template retrieved. Finding evaluated via contract asymmetry analysis.")
                            
                    st.markdown(f"**Confidence Score:** `{r.get('confidence', 0.85):.2f}` | **Risk Identifier:** `{r.get('risk_id')}`")

    # TAB 4: PLAIN-LANGUAGE SUMMARY
    with tabs[3]:
        st.markdown("### 💡 Plain-Language Breakdown for Non-Lawyers")
        risk_explanations = final_state.get("risk_explanations", []) or []
        if risk_explanations:
            for exp_item in risk_explanations:
                st.markdown(f"""
                <div class="risk-card" style="border-left: 5px solid #3B82F6;">
                    <div style="font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem;">{exp_item.get('risk_id', 'Risk Item').upper()} ({exp_item.get('severity', 'Notice')})</div>
                    <p><strong>What does the clause say?</strong><br/>{exp_item.get('clause_summary', '')}</p>
                    <p><strong>Why does this matter financially/operationally?</strong><br/>{exp_item.get('business_impact', '')}</p>
                    <p><strong>What action or negotiation point is recommended?</strong><br/>{exp_item.get('review_action', '')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write(final_state.get("simplified_explanation", "No plain-language breakdown available."))

    # TAB 5: HUMAN DECISIONS
    with tabs[4]:
        st.markdown("### 🧑‍⚖️ Human Review Decision Audit Trail")
        h_reviews = final_state.get("human_reviews", []) or []
        if not h_reviews:
            st.info("No human review actions recorded for this review.")
        else:
            for h in h_reviews:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #E2E8F0; padding: 1rem 1.25rem; border-radius: 10px; margin-bottom: 0.75rem;">
                    <strong>Risk ID:</strong> {h.get('risk_id')} | 
                    <strong>Decision:</strong> <span style="font-weight: 800; color: #1E40AF;">{h.get('decision')}</span> | 
                    <strong>Timestamp:</strong> {h.get('timestamp', 'N/A')}<br/>
                    <strong>Reviewer Notes:</strong> {h.get('note') or 'None'}
                </div>
                """, unsafe_allow_html=True)

    # TAB 6: RAW DATA
    with tabs[5]:
        st.markdown("### 📊 Raw Multi-Agent State")
        st.json(final_state)


def render_comparison_dashboard(comp_results: dict):
    comparisons = comp_results.get("comparisons", []) or []
    
    st.markdown("---")
    st.markdown("### 🔍 Semantic Redline & Revision Analysis")
    st.caption("Semantic clause alignment and risk impact assessment between Version A and Version B.")
    
    if not comparisons:
        st.info("No clause differences or modifications detected between the two document versions.")
        return

    inc_count = sum(1 for c in comparisons if "increase" in str(c.get("risk_impact", "")).lower())
    dec_count = sum(1 for c in comparisons if "decrease" in str(c.get("risk_impact", "")).lower())
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Total Revisions</div>
            <div class="metric-val">{len(comparisons)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Risk Increased</div>
            <div class="metric-val" style="color: #DC2626;">{inc_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Risk Mitigated / Decreased</div>
            <div class="metric-val" style="color: #059669;">{dec_count}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for idx, item in enumerate(comparisons, start=1):
        status = item.get("status", "MODIFIED")
        risk_imp = item.get("risk_impact", "No Change")
        c_type = item.get("clause_type", "General Provision")
        
        status_colors = {"ADDED": "#059669", "REMOVED": "#DC2626", "MODIFIED": "#D97706", "UNCHANGED": "#64748B"}
        s_color = status_colors.get(status, "#1E3A8A")
        
        st.markdown(f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; margin-bottom: 1.5rem; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="background: #F8FAFC; border-bottom: 1px solid #E2E8F0; padding: 0.85rem 1.35rem; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 800; color: #0F172A; font-size: 1rem;">#{idx}. {c_type}</span>
                <div>
                    <span style="background: {s_color}22; color: {s_color}; border: 1px solid {s_color}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
                        {status}
                    </span>
                    <span style="background: #EEF2F6; color: #1E293B; border: 1px solid #CBD5E1; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; margin-left: 0.5rem;">
                        Impact: {risk_imp}
                    </span>
                </div>
            </div>
            <div style="padding: 1.35rem;">
                <div class="diff-container">
                    <div class="diff-a">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #991B1B; text-transform: uppercase; margin-bottom: 0.4rem;">Original (Version A)</div>
                        <div style="white-space: pre-wrap;">{item.get('old_text') or '[None - Newly Added in Version B]'}</div>
                    </div>
                    <div class="diff-b">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #166534; text-transform: uppercase; margin-bottom: 0.4rem;">Modified (Version B)</div>
                        <div style="white-space: pre-wrap;">{item.get('new_text') or '[None - Deleted in Version B]'}</div>
                    </div>
                </div>
                <div style="border-top: 1px solid #F1F5F9; padding-top: 0.75rem; font-size: 0.92rem; color: #334155;">
                    <strong>Change Summary:</strong> {item.get('change_summary', '')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================

with st.sidebar:
    st.markdown("### ⚖️ LegalValidate Engine")
    
    env_api_key = os.getenv("GROQ_API_KEY")
    if not env_api_key or env_api_key == "your_groq_api_key_here":
        st.error("⚠️ GROQ_API_KEY Missing")
        st.info("Set your Groq key in `.env` to activate the engine.")
        api_key = None
    else:
        st.success("✅ Engine Active (Groq)")
        api_key = env_api_key
        
    st.markdown("---")
    st.markdown("### 🛠️ Mode Selection")
    app_mode = st.selectbox(
        "Choose Mode:",
        ["Single Document Review", "Double Document Comparison", "Audit History"],
        key="app_mode"
    )
    
    # PDF Download in Sidebar if single review completed
    if app_mode == "Single Document Review" and st.session_state.get("final_state"):
        st.markdown("---")
        st.markdown("### 📥 Export Report")
        try:
            pdf_bytes = generate_pdf_report(st.session_state.final_state)
            fname_clean = st.session_state.final_state.get('document_type', 'Contract').replace(' ', '_')
            st.download_button(
                label="📥 Download PDF Audit Report",
                data=pdf_bytes,
                file_name=f"LegalValidate_Report_{fname_clean}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as err:
            st.warning(f"PDF export unavailable: {err}")
            
    st.markdown("---")
    st.caption("🛡️ **Notice**: Analytical AI assistant. Does not constitute formal legal counsel.")


# =====================================================================
# MAIN HEADER & LEGAL SAFETY BANNER
# =====================================================================

st.markdown("""
<div class="header-banner">
    <h1>⚖️ LegalValidate AI</h1>
    <p>Multi-Agent Legal Document Validation, RAG Grounding & Risk Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <span>⚠️ <strong>LEGAL NOTICE:</strong> LegalValidate AI is an analytical assistant for document review and risk detection. It does not provide formal legal advice or substitute for review by qualified legal counsel.</span>
</div>
""", unsafe_allow_html=True)

# Handle mode transitions cleanly
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = app_mode
if st.session_state.prev_mode != app_mode:
    st.session_state.prev_mode = app_mode
    st.session_state.single_status = 'idle'
    st.session_state.final_state = None
    st.session_state.comp_status = 'idle'
    st.session_state.comp_results = None


# =====================================================================
# MODE 1: SINGLE DOCUMENT REVIEW
# =====================================================================

if app_mode == "Single Document Review":
    
    # Sample Selector or Upload
    col_top1, col_top2 = st.columns([1, 1])
    with col_top1:
        st.markdown("#### 📂 Upload Document")
        uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], key="single_file_up")
        
    with col_top2:
        st.markdown("#### ⚡ Or Select Sample Contract")
        sample_choice = st.selectbox("Choose a sample to test instantly:", list(SAMPLE_DOCUMENTS.keys()), key="sample_selector")
        if sample_choice and SAMPLE_DOCUMENTS[sample_choice]:
            if st.button("🚀 Load Selected Sample Contract", use_container_width=True):
                st.session_state.doc_text = clean_text(SAMPLE_DOCUMENTS[sample_choice])
                st.session_state.active_doc_name = sample_choice.split("(")[0].strip() + ".txt"
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.single_status = 'running_pipeline'
                st.session_state.final_state = None
                st.rerun()

    # If file uploaded
    if uploaded_file and api_key:
        file_bytes = uploaded_file.read()
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.get("active_file_id") != file_id:
            st.session_state.active_file_id = file_id
            st.session_state.active_doc_name = uploaded_file.name
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.single_status = 'idle'
            st.session_state.final_state = None

        if st.sidebar.button("🔄 Reset Review", use_container_width=True):
            st.session_state.single_status = 'idle'
            st.session_state.final_state = None
            st.rerun()

        if st.session_state.get("single_status", "idle") == 'idle':
            st.markdown("---")
            col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
            with col_b2:
                if st.button("🚀 Run Multi-Agent Validation", use_container_width=True):
                    with st.spinner("Extracting document content..."):
                        raw_text, _ = extract_document_text(file_bytes, uploaded_file.name)
                        st.session_state.doc_text = clean_text(raw_text)
                        st.session_state.single_status = 'running_pipeline'
                        st.rerun()

    # EXECUTION PIPELINE
    if st.session_state.get("single_status") == 'running_pipeline' and api_key:
        status_box = st.status("🧠 Multi-Agent Orchestration in Progress...", expanded=True)
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            app = create_graph()
            
            doc_name = st.session_state.get("active_doc_name", "document.txt")
            initial_state = {
                "text": st.session_state.doc_text,
                "api_key": api_key,
                "file_metadata": {"filename": doc_name},
                "is_legal": None,
                "document_type": None,
                "classification_reason": None,
                "classification_confidence": None,
                "problematic_explanation": None,
                "document_analysis": None,
                "key_clauses": [],
                "risks": [],
                "critic_reviews": [],
                "summary": None,
                "simplified_explanation": None,
                "risk_explanations": [],
                "human_reviews": [],
                "final_report": None
            }
            
            current_state = initial_state.copy()
            status_box.write("🔍 **Legal Classifier**: Evaluating legal validity and document type...")
            
            for output in app.stream(initial_state, config):
                if isinstance(output, dict):
                    for node_name, node_state in output.items():
                        if node_name == "classifier":
                            is_leg = node_state.get("is_legal")
                            status_box.write(f"✅ **Classifier**: {'Legal Contract' if is_leg else 'Non-Legal Document'} ({node_state.get('document_type')})")
                        elif node_name == "analyzer":
                            status_box.write("✅ **Document Analyzer**: Extracted structure, dates & obligations.")
                        elif node_name == "risk_detector":
                            status_box.write("✅ **Risk Detector**: RAG baseline comparison & risk detection complete.")
                        elif node_name == "critic":
                            status_box.write("✅ **Critic Agent**: Audited and verified risk severities.")
                        elif node_name == "explainer":
                            status_box.write("✅ **Explanation Agent**: Plain-language summaries compiled.")
                        elif node_name == "non_legal_explainer":
                            status_box.write("ℹ️ **Non-Legal Explainer**: Early exit completed.")
                        if isinstance(node_state, dict):
                            current_state.update(node_state)

            st.session_state.final_state = current_state
            
            if current_state.get("is_legal") is True:
                st.session_state.single_status = 'hitl_review'
                status_box.update(label="⏸️ Paused at Human-in-the-Loop Review Gate", state="complete", expanded=False)
            else:
                st.session_state.single_status = 'completed'
                save_analysis(doc_name, st.session_state.final_state)
                status_box.update(label="✅ Analysis Complete", state="complete", expanded=False)
                
            st.rerun()
        except Exception as e:
            st.session_state.single_status = 'idle'
            status_box.update(label="❌ Analysis Failed", state="error")
            st.error(f"Orchestration Error: {str(e)}")

    # HITL INTERACTIVE REVIEW WORKFLOW
    elif st.session_state.get("single_status") == 'hitl_review':
        st.markdown("---")
        st.markdown("### 🧑‍⚖️ Human-in-the-Loop Review Checkpoint")
        st.info("The AI Risk Detector and Critic have verified candidate contract risks. Review each finding below, select your decision (**Accept Risk**, **Reject Finding**, or **Mark for Review**), add reviewer notes, and approve to compile the final report.")
        
        risks = st.session_state.final_state.get("risks", []) or []
        critic_reviews = st.session_state.final_state.get("critic_reviews", []) or []
        critic_map = {r.get("risk_id"): r for r in critic_reviews if isinstance(r, dict)}
        
        with st.form("hitl_review_form"):
            for idx, r in enumerate(risks, start=1):
                if not isinstance(r, dict):
                    continue
                r_id = r.get("risk_id", f"risk_{idx}")
                c_rev = critic_map.get(r_id, {})
                c_sev = c_rev.get("verified_severity", r.get("severity", "MEDIUM"))
                c_valid = c_rev.get("is_valid", True)
                
                st.markdown(f"""
                <div class="hitl-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: #0F172A;">#{idx}. {r.get('risk_type')}</span>
                        <div>
                            <span class="badge badge-{c_sev.lower()}">{c_sev}</span>
                            <span class="badge {'badge-valid' if c_valid else 'badge-removed'}">Critic: {'Verified' if c_valid else 'Filtered'}</span>
                        </div>
                    </div>
                    <div class="clause-box">"{r.get('clause')}"</div>
                    <div style="font-size: 0.9rem; color: #334155; margin-bottom: 0.75rem;"><strong>Analysis:</strong> {r.get('explanation')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                c_act1, c_act2 = st.columns([1, 2])
                with c_act1:
                    st.radio(
                        f"Review Decision for #{idx}:",
                        ["Accept Risk", "Reject Finding", "Mark for Review"],
                        key=f"hitl_dec_{r_id}"
                    )
                with c_act2:
                    st.text_input(
                        f"Notes for #{idx}:",
                        placeholder="e.g. Request counterparty to cap liability at 12 months fees",
                        key=f"hitl_note_{r_id}"
                    )
                st.markdown("<br/>", unsafe_allow_html=True)

            submitted = st.form_submit_button("✅ Approve & Compile Final Audit Report", use_container_width=True)
            if submitted:
                decisions_list = []
                for idx, r in enumerate(risks, start=1):
                    if isinstance(r, dict):
                        rid = r.get("risk_id", f"risk_{idx}")
                        dec_val = st.session_state.get(f"hitl_dec_{rid}", "Accept Risk")
                        note_val = st.session_state.get(f"hitl_note_{rid}", "")
                        decisions_list.append({
                            "risk_id": rid,
                            "decision": dec_val,
                            "note": note_val,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                st.session_state.final_state["human_reviews"] = decisions_list
                st.session_state.single_status = 'completed'
                
                # Persist to database
                doc_name = st.session_state.get("active_doc_name", "document.pdf")
                save_analysis(doc_name, st.session_state.final_state)
                st.rerun()

    # COMPLETED DASHBOARD
    elif st.session_state.get("single_status") == 'completed':
        st.markdown("---")
        render_single_analysis_dashboard(st.session_state.final_state)


# =====================================================================
# MODE 2: DOUBLE DOCUMENT COMPARISON
# =====================================================================

elif app_mode == "Double Document Comparison":
    st.markdown("### 🔍 Semantic Redline & Comparison Review")
    st.caption("Upload two versions of a contract to align clauses, detect additions/deletions/modifications, and evaluate risk impact.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        file_a = st.file_uploader("📂 Upload Original Contract (Version A)", type=["pdf", "docx", "txt"], key="comp_uploader_a")
    with col_c2:
        file_b = st.file_uploader("📂 Upload Modified Contract (Version B)", type=["pdf", "docx", "txt"], key="comp_uploader_b")

    # Quick demo redline button
    if st.button("⚡ Load Sample Revision Comparison (Lease V1 vs V2 with Escalations)"):
        st.session_state.txt_a = """COMMERCIAL LEASE.
1. Payment: Rent is $5,000/month, payable on the 1st of each month.
2. Security Deposit: Landlord shall return the security deposit within 30 days of lease expiration minus documented damages.
3. Entry: Landlord may enter the premises during business hours upon providing 24 hours prior written notice."""
        st.session_state.txt_b = """COMMERCIAL LEASE.
1. Payment: Rent is $5,000/month. If payment is delayed past 3 days, a 25% late penalty applies.
2. Security Deposit: Landlord shall unconditionally retain the full security deposit upon termination regardless of condition.
3. Entry: Landlord reserves the right to enter the premises at any hour without prior notice."""
        st.session_state.comp_name_a = "Lease_Original_V1.txt"
        st.session_state.comp_name_b = "Lease_Modified_V2.txt"
        st.session_state.comp_thread_id = str(uuid.uuid4())
        st.session_state.comp_status = 'running'
        st.rerun()

    if file_a and file_b and api_key:
        bytes_a = file_a.read()
        bytes_b = file_b.read()
        comp_id = f"{file_a.name}_{file_a.size}_{file_b.name}_{file_b.size}"
        
        if st.session_state.get("active_comp_id") != comp_id:
            st.session_state.active_comp_id = comp_id
            st.session_state.comp_name_a = file_a.name
            st.session_state.comp_name_b = file_b.name
            st.session_state.comp_thread_id = str(uuid.uuid4())
            st.session_state.comp_status = 'idle'
            st.session_state.comp_results = None

        if st.sidebar.button("🔄 Reset Comparison", use_container_width=True):
            st.session_state.comp_status = 'idle'
            st.session_state.comp_results = None
            st.rerun()

        if st.session_state.get("comp_status", "idle") == 'idle':
            st.markdown("---")
            col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
            with col_b2:
                if st.button("🚀 Compare Document Versions", use_container_width=True):
                    with st.spinner("Extracting text and aligning clauses..."):
                        t_a, _ = extract_document_text(bytes_a, file_a.name)
                        t_b, _ = extract_document_text(bytes_b, file_b.name)
                        st.session_state.txt_a = clean_text(t_a)
                        st.session_state.txt_b = clean_text(t_b)
                        st.session_state.comp_status = 'running'
                        st.rerun()

    if st.session_state.get("comp_status") == 'running' and api_key:
        status_box = st.status("🧠 Semantic Clause Alignment & Redlining in Progress...", expanded=True)
        try:
            comp_config = {"configurable": {"thread_id": st.session_state.comp_thread_id}}
            init_comp = {
                "text_a": st.session_state.txt_a,
                "text_b": st.session_state.txt_b,
                "api_key": api_key,
                "comparisons": []
            }
            
            final_comp = init_comp.copy()
            for output in comparison_app.stream(init_comp, comp_config):
                if isinstance(output, dict):
                    for node_name, node_state in output.items():
                        if node_name == "compare":
                            status_box.write("✅ **Comparator Agent**: Semantic alignment & risk direction complete.")
                        if isinstance(node_state, dict):
                            final_comp.update(node_state)
                    
            st.session_state.comp_results = final_comp
            st.session_state.comp_status = 'completed'
            
            # Save to SQLite
            name_a = st.session_state.get("comp_name_a", "Version_A.pdf")
            name_b = st.session_state.get("comp_name_b", "Version_B.pdf")
            save_comparison(name_a, name_b, final_comp)
            
            status_box.update(label="✅ Comparison Complete", state="complete", expanded=False)
            st.rerun()
        except Exception as e:
            st.session_state.comp_status = 'idle'
            status_box.update(label="❌ Comparison Failed", state="error")
            st.error(f"Comparison Error: {str(e)}")

    elif st.session_state.get("comp_status") == 'completed':
        render_comparison_dashboard(st.session_state.comp_results)


# =====================================================================
# MODE 3: AUDIT HISTORY
# =====================================================================

elif app_mode == "Audit History":
    st.markdown("## 📜 Analysis & Comparison Audit History")
    st.caption("Inspect past multi-agent reviews and redlines with full audit trails.")
    
    t_single, t_comp = st.tabs(["📄 Document Reviews", "🔍 Document Comparisons"])
    
    with t_single:
        analyses = get_all_analyses()
        if not analyses:
            st.info("No past document reviews stored in the database.")
        else:
            options = {
                f"#{item['id']} | {item['filename']} ({item['timestamp']}) - {item['document_type']} [{item['classification']}]": item['id']
                for item in analyses
            }
            selected_label = st.selectbox("Select a past review:", list(options.keys()))
            if selected_label:
                selected_id = options[selected_label]
                record = get_analysis_by_id(selected_id)
                if record:
                    st.markdown("---")
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"### 📂 Review #{record['id']}: {record['filename']}")
                        st.caption(f"**Saved Date:** {record['timestamp']}")
                    with col_h2:
                        try:
                            pdf_data = generate_pdf_report(record["full_result"])
                            st.download_button(
                                label="📥 Export PDF Report",
                                data=pdf_data,
                                file_name=f"LegalValidate_Report_DB{record['id']}_{record['filename'].replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as err:
                            st.warning(f"PDF unavailable: {err}")
                            
                    st.markdown("---")
                    render_single_analysis_dashboard(record["full_result"])
                    
    with t_comp:
        comparisons_list = get_all_comparisons()
        if not comparisons_list:
            st.info("No past document comparisons stored in the database.")
        else:
            comp_opts = {
                f"#{item['id']} | {item['filename_a']} vs {item['filename_b']} ({item['timestamp']}) - {item['comparison_count']} revisions": item['id']
                for item in comparisons_list
            }
            selected_comp_lbl = st.selectbox("Select a past comparison:", list(comp_opts.keys()))
            if selected_comp_lbl:
                selected_comp_id = comp_opts[selected_comp_lbl]
                comp_rec = get_comparison_by_id(selected_comp_id)
                if comp_rec:
                    st.markdown("---")
                    st.markdown(f"### 📂 Comparison #{comp_rec['id']}: {comp_rec['filename_a']} vs {comp_rec['filename_b']}")
                    st.caption(f"**Date:** {comp_rec['timestamp']}")
                    st.markdown("---")
                    render_comparison_dashboard(comp_rec["full_result"])
