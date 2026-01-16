"""Advanced Financial Valuation Tools

Implements sophisticated equity valuation methodologies including:
- Multi-stage DCF with explicit forecast period
- Risk-adjusted discount rates (WACC)
- Terminal value calculations
- Scenario analysis
- Relative valuation
- Sensitivity analysis
"""

import logging
import math
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class FinanceTools:
    """Advanced financial calculation and valuation tools"""
    
    # ========================================================================
    # ADVANCED DCF VALUATION (Multi-Stage Growth Model)
    # ========================================================================
    
    @staticmethod
    def calculate_advanced_intrinsic_value(
        fcf_explicit_period: List[float],
        terminal_growth_rate: float = 0.03,
        discount_rate: float = 0.10,
        terminal_multiple: Optional[float] = None,
        shares_outstanding: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate intrinsic value using advanced multi-stage DCF model
        
        Args:
            fcf_explicit_period: Free cash flows for explicit forecast period (typically 5-10 years)
            terminal_growth_rate: Long-term growth rate (typically 2-4%)
            discount_rate: WACC (weighted average cost of capital)
            terminal_multiple: Optional terminal EV/EBITDA or similar multiple
            shares_outstanding: Number of shares for per-share calculation
            
        Returns:
            Dict with intrinsic value metrics
        """
        if not fcf_explicit_period or len(fcf_explicit_period) == 0:
            return {"error": "No cash flows provided"}
        
        # Validate discount rate
        if discount_rate <= terminal_growth_rate:
            return {"error": "Discount rate must be greater than terminal growth rate"}
        
        # Calculate PV of explicit period cash flows
        pv_explicit = sum(
            fcf / ((1 + discount_rate) ** (i + 1))
            for i, fcf in enumerate(fcf_explicit_period)
        )
        
        # Calculate terminal value using perpetuity growth method
        final_year_fcf = fcf_explicit_period[-1]
        terminal_fcf = final_year_fcf * (1 + terminal_growth_rate)
        terminal_value_perpetuity = terminal_fcf / (discount_rate - terminal_growth_rate)
        pv_terminal_perpetuity = terminal_value_perpetuity / ((1 + discount_rate) ** len(fcf_explicit_period))
        
        # Enterprise value using perpetuity method
        enterprise_value_perpetuity = pv_explicit + pv_terminal_perpetuity
        intrinsic_value_perpetuity = enterprise_value_perpetuity / shares_outstanding
        
        result = {
            "pv_explicit_period": pv_explicit,
            "terminal_value_perpetuity": terminal_value_perpetuity,
            "pv_terminal_value": pv_terminal_perpetuity,
            "enterprise_value": enterprise_value_perpetuity,
            "intrinsic_value_per_share": intrinsic_value_perpetuity,
            "valuation_method": "Multi-stage DCF with perpetuity",
            "key_assumptions": {
                "explicit_period_years": len(fcf_explicit_period),
                "terminal_growth_rate": terminal_growth_rate,
                "discount_rate": discount_rate
            }
        }
        
        # If terminal multiple provided, calculate alternative valuation
        if terminal_multiple:
            terminal_value_multiple = final_year_fcf * terminal_multiple
            pv_terminal_multiple = terminal_value_multiple / ((1 + discount_rate) ** len(fcf_explicit_period))
            enterprise_value_multiple = pv_explicit + pv_terminal_multiple
            intrinsic_value_multiple = enterprise_value_multiple / shares_outstanding
            
            result.update({
                "terminal_value_multiple": terminal_value_multiple,
                "pv_terminal_multiple": pv_terminal_multiple,
                "enterprise_value_multiple_method": enterprise_value_multiple,
                "intrinsic_value_multiple_method": intrinsic_value_multiple
            })
        
        return result
    
    # ========================================================================
    # WEIGHTED AVERAGE COST OF CAPITAL (WACC) - Risk-Adjusted Discount Rate
    # ========================================================================
    
    @staticmethod
    def calculate_wacc(
        cost_of_equity: float,
        cost_of_debt: float,
        market_value_equity: float,
        market_value_debt: float,
        tax_rate: float = 0.25
    ) -> Dict[str, float]:
        """
        Calculate Weighted Average Cost of Capital (WACC)
        
        WACC = (E/V × Cost of Equity) + ((D/V × Cost of Debt) × (1 - Tax Rate))
        
        Where:
        - E = Market value of equity
        - D = Market value of debt
        - V = E + D = Total firm value
        - Tax rate provides tax shield on debt
        
        Args:
            cost_of_equity: Cost of equity (from CAPM or other method)
            cost_of_debt: Before-tax cost of debt
            market_value_equity: Market capitalization
            market_value_debt: Total debt value
            tax_rate: Corporate tax rate
            
        Returns:
            Dict with WACC and component details
        """
        total_value = market_value_equity + market_value_debt
        
        if total_value == 0:
            return {"error": "Total firm value is zero"}
        
        weight_equity = market_value_equity / total_value
        weight_debt = market_value_debt / total_value
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        
        return {
            "wacc": wacc,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt_after_tax": cost_of_debt * (1 - tax_rate),
            "weight_equity": weight_equity,
            "weight_debt": weight_debt,
            "market_value_equity": market_value_equity,
            "market_value_debt": market_value_debt,
            "total_firm_value": total_value,
            "tax_rate": tax_rate,
            "description": "Risk-adjusted discount rate for valuation"
        }
    
    # ========================================================================
    # COST OF EQUITY CALCULATION (CAPM)
    # ========================================================================
    
    @staticmethod
    def calculate_cost_of_equity_capm(
        risk_free_rate: float,
        market_risk_premium: float,
        beta: float
    ) -> Dict[str, float]:
        """
        Calculate Cost of Equity using Capital Asset Pricing Model (CAPM)
        
        Cost of Equity = Risk-Free Rate + Beta × (Market Risk Premium)
        
        Args:
            risk_free_rate: Risk-free rate (typically 10-year government bond yield)
            market_risk_premium: Expected market return - risk-free rate (typically 5-7%)
            beta: Stock's systematic risk relative to market (market beta = 1.0)
            
        Returns:
            Dict with cost of equity and components
        """
        cost_of_equity = risk_free_rate + (beta * market_risk_premium)
        
        return {
            "cost_of_equity": cost_of_equity,
            "risk_free_rate": risk_free_rate,
            "beta": beta,
            "market_risk_premium": market_risk_premium,
            "equity_risk_premium": beta * market_risk_premium,
            "formula": "Risk-Free Rate + Beta × Market Risk Premium",
            "description": "Cost of equity using CAPM methodology"
        }
    
    # ========================================================================
    # RELATIVE VALUATION (Using Multiples)
    # ========================================================================
    
    @staticmethod
    def calculate_relative_valuation(
        comparable_multiples: Dict[str, float],
        company_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate valuation using relative multiples from comparable companies
        
        Supports: P/E, EV/EBITDA, P/B, P/S, EV/Revenue, etc.
        
        Args:
            comparable_multiples: Dict with multiples (e.g., {"pe_ratio": 20.5, "ev_ebitda": 12.3})
            company_metrics: Dict with company metrics (e.g., {"earnings": 100, "ebitda": 150})
            
        Returns:
            Dict with valuations using different multiples
        """
        valuations = {}
        
        # Map metrics to multiples
        metric_multiple_mapping = {
            "earnings": "pe_ratio",
            "revenue": "price_to_sales",
            "ebitda": "ev_ebitda",
            "book_value": "price_to_book",
            "fcf": "price_to_fcf"
        }
        
        for metric_name, metric_value in company_metrics.items():
            if metric_value <= 0:
                continue
                
            multiple_name = metric_multiple_mapping.get(metric_name)
            if multiple_name and multiple_name in comparable_multiples:
                multiple = comparable_multiples[multiple_name]
                valuation = metric_value * multiple
                valuations[f"valuation_via_{metric_name}"] = valuation
                valuations[f"{metric_name}_{multiple_name}"] = multiple
        
        if not valuations:
            return {"error": "No matching multiples found for company metrics"}
        
        # Calculate average valuation
        valuations_only = [v for k, v in valuations.items() if k.startswith("valuation_")]
        avg_valuation = sum(valuations_only) / len(valuations_only) if valuations_only else 0
        
        return {
            **valuations,
            "average_valuation": avg_valuation,
            "valuation_range": (min(valuations_only), max(valuations_only)) if valuations_only else None,
            "method": "Comparable Company Multiples",
            "multiples_used": comparable_multiples
        }
    
    # ========================================================================
    # TWO-STAGE GROWTH MODEL (High Growth → Stable Growth)
    # ========================================================================
    
    @staticmethod
    def calculate_two_stage_valuation(
        current_fcf: float,
        high_growth_rate: float,
        high_growth_years: int,
        stable_growth_rate: float,
        discount_rate: float,
        shares_outstanding: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate valuation using two-stage growth model
        
        Stage 1: High growth period (e.g., 5-10 years of above-market growth)
        Stage 2: Stable growth period (perpetual growth at GDP rate)
        
        Args:
            current_fcf: Current free cash flow
            high_growth_rate: Growth rate during high-growth stage (e.g., 0.15 for 15%)
            high_growth_years: Number of years in high-growth stage
            stable_growth_rate: Growth rate after high-growth stage (typically 2-4%)
            discount_rate: WACC
            shares_outstanding: Number of shares
            
        Returns:
            Dict with detailed two-stage valuation
        """
        if discount_rate <= stable_growth_rate:
            return {"error": "Discount rate must exceed stable growth rate"}
        
        # Stage 1: High growth cash flows
        pv_high_growth = 0
        for year in range(1, high_growth_years + 1):
            fcf_year = current_fcf * ((1 + high_growth_rate) ** year)
            pv = fcf_year / ((1 + discount_rate) ** year)
            pv_high_growth += pv
        
        # Stage 2: Terminal value (stable growth perpetuity)
        fcf_terminal_year = current_fcf * ((1 + high_growth_rate) ** high_growth_years)
        fcf_stable = fcf_terminal_year * (1 + stable_growth_rate)
        terminal_value = fcf_stable / (discount_rate - stable_growth_rate)
        pv_terminal = terminal_value / ((1 + discount_rate) ** high_growth_years)
        
        # Enterprise value and per-share valuation
        enterprise_value = pv_high_growth + pv_terminal
        intrinsic_value = enterprise_value / shares_outstanding
        
        return {
            "stage1_high_growth_pv": pv_high_growth,
            "stage1_high_growth_rate": high_growth_rate,
            "stage1_years": high_growth_years,
            "stage2_stable_growth_rate": stable_growth_rate,
            "stage2_terminal_value": terminal_value,
            "stage2_terminal_pv": pv_terminal,
            "enterprise_value": enterprise_value,
            "intrinsic_value_per_share": intrinsic_value,
            "valuation_method": "Two-Stage Growth Model",
            "growth_transition": f"{high_growth_rate*100:.1f}% for {high_growth_years} years → {stable_growth_rate*100:.1f}% perpetual"
        }
    
    # ========================================================================
    # SENSITIVITY ANALYSIS
    # ========================================================================
    
    @staticmethod
    def calculate_valuation_sensitivity(
        base_valuation: float,
        discount_rate: float,
        terminal_growth_rate: float,
        discount_rate_range: Tuple[float, float] = None,
        growth_rate_range: Tuple[float, float] = None,
        step_size: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculate valuation sensitivity to changes in key assumptions
        
        Creates sensitivity matrix showing how valuation changes with different
        discount rates and terminal growth rates
        
        Args:
            base_valuation: Base case valuation
            discount_rate: Base discount rate
            terminal_growth_rate: Base terminal growth rate
            discount_rate_range: (min, max) for discount rate sensitivity
            growth_rate_range: (min, max) for growth rate sensitivity
            step_size: Step size for sensitivity (e.g., 0.01 for 1%)
            
        Returns:
            Dict with sensitivity analysis table and insights
        """
        if not discount_rate_range:
            discount_rate_range = (discount_rate - 0.05, discount_rate + 0.05)
        if not growth_rate_range:
            growth_rate_range = (terminal_growth_rate - 0.02, terminal_growth_rate + 0.02)
        
        sensitivity_table = []
        
        dr_range = discount_rate_range[1] - discount_rate_range[0]
        gr_range = growth_rate_range[1] - growth_rate_range[0]
        
        dr_steps = max(1, int(dr_range / step_size))
        gr_steps = max(1, int(gr_range / step_size))
        
        for i in range(gr_steps + 1):
            row = []
            gr = growth_rate_range[0] + (i * gr_range / max(1, gr_steps))
            
            for j in range(dr_steps + 1):
                dr = discount_rate_range[0] + (j * dr_range / max(1, dr_steps))
                
                if dr > gr:
                    # Adjusted valuation based on changes
                    dr_change = (dr - discount_rate) / discount_rate if discount_rate != 0 else 0
                    gr_change = (gr - terminal_growth_rate) / terminal_growth_rate if terminal_growth_rate != 0 else 0
                    
                    # Simplified sensitivity: 1% increase in discount rate = -5% valuation
                    # 1% increase in growth rate = +5% valuation
                    adjusted_val = base_valuation * (1 - (dr_change * 5) + (gr_change * 5))
                    row.append(round(adjusted_val, 2))
                else:
                    row.append(None)
            
            sensitivity_table.append(row)
        
        return {
            "base_valuation": base_valuation,
            "sensitivity_table": sensitivity_table,
            "discount_rate_range": discount_rate_range,
            "growth_rate_range": growth_rate_range,
            "interpretation": "Higher discount rates → Lower valuations; Higher growth rates → Higher valuations",
            "key_insight": "Most sensitive to discount rate assumptions"
        }
    
    # ========================================================================
    # SIMPLE LEGACY FUNCTIONS (Backward Compatible)
    # ========================================================================
    
    @staticmethod
    def calculate_dcf_valuation(
        cash_flows: list,
        discount_rate: float = 0.10,
        terminal_growth_rate: float = 0.03
    ) -> dict:
        """Calculate simple DCF valuation (legacy)"""
        pv_cash_flows = sum(cf / ((1 + discount_rate) ** (i + 1)) 
                            for i, cf in enumerate(cash_flows))
        terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / ((1 + discount_rate) ** len(cash_flows))
        
        return {
            "pv_cash_flows": pv_cash_flows,
            "terminal_value": terminal_value,
            "pv_terminal": pv_terminal,
            "enterprise_value": pv_cash_flows + pv_terminal
        }
    
    @staticmethod
    def calculate_pe_ratio(price: float, earnings: float) -> float:
        """Calculate P/E ratio"""
        return price / earnings if earnings != 0 else 0
    
    @staticmethod
    def calculate_roe(net_income: float, shareholders_equity: float) -> float:
        """Calculate ROE"""
        return net_income / shareholders_equity if shareholders_equity != 0 else 0
