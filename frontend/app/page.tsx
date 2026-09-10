"use client";

import React, { useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  HelpCircle,
  RotateCcw,
  ShieldAlert,
} from "lucide-react";

interface ReasonCode {
  feature: string;
  impact: number;
  direction: "increases cost" | "decreases cost";
}

interface PredictionResponse {
  predicted_cost: number;
  interval_80: [number, number];
  premium_tier: "Standard" | "Loaded" | "Refer to Underwriter";
  reason_codes: ReasonCode[];
  model_version: string;
  decision_timestamp: string;
}

interface FormState {
  age: number;
  bmi: number;
  children: number;
  smoker: "no" | "yes";
  region: "northeast" | "northwest" | "southeast" | "southwest";
}

const DEFAULT_FORM: FormState = {
  age: 35,
  bmi: 26.5,
  children: 0,
  smoker: "no",
  region: "southwest",
};

export default function MedicalCostPredictor() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiUrl =
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";
  const apiKey =
    process.env.NEXT_PUBLIC_API_KEY || "dev-api-key-change-in-production";

  const handleChange = (
    field: keyof FormState,
    value: string | number
  ) => {
    setForm((prev) => ({
      ...prev,
      [field]:
        field === "age" || field === "children"
          ? parseInt(String(value), 10) || 0
          : field === "bmi"
          ? parseFloat(String(value)) || 0
          : value,
    }));
  };

  const handleReset = () => {
    setForm(DEFAULT_FORM);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Client-side quick bounds validation
    if (form.age < 18 || form.age > 100) {
      setError("Age must be between 18 and 100 years.");
      setLoading(false);
      return;
    }
    if (form.bmi < 10.0 || form.bmi > 60.0) {
      setError("BMI must be between 10.0 and 60.0 kg/m².");
      setLoading(false);
      return;
    }
    if (form.children < 0 || form.children > 20) {
      setError("Number of dependents must be between 0 and 20.");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": apiKey,
        },
        body: JSON.stringify({
          age: Number(form.age),
          bmi: Number(form.bmi),
          children: Number(form.children),
          smoker: form.smoker,
          region: form.region,
        }),
      });

      if (!response.ok) {
        if (response.status === 403) {
          throw new Error("API access unauthorized. Invalid or missing API key.");
        }
        if (response.status === 422) {
          const detail = await response.json();
          const msg = Array.isArray(detail?.detail)
            ? detail.detail.map((d: { msg?: string }) => d.msg).join(", ")
            : "Invalid input attributes supplied.";
          throw new Error(`Validation Error: ${msg}`);
        }
        if (response.status === 503) {
          throw new Error("Prediction model is still initializing or not loaded.");
        }
        throw new Error(`Prediction request failed with status code ${response.status}.`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setError(
          `Unable to connect to the prediction API at ${apiUrl}. Please verify the backend service is running.`
        );
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred while predicting costs.");
      }
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(val);

  return (
    <div className="min-h-screen py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <header className="text-center mb-10">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Medical Cost & Risk Predictor
          </h1>
          <p className="mt-3 text-base sm:text-lg text-slate-600 max-w-2xl mx-auto">
            Estimate expected annual medical expenditures, calibrated 80% prediction
            intervals, and underwriting risk tiers in real time.
          </p>
        </header>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Form Column */}
          <div className="lg:col-span-6 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8">
            <h2 className="text-lg font-bold text-slate-900 mb-1">
              Applicant Information
            </h2>
            <p className="text-sm text-slate-500 mb-6">
              Enter individual demographic and lifestyle factors.
            </p>

            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Age */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label htmlFor="age" className="text-sm font-medium text-slate-700">
                    Age
                  </label>
                  <span className="text-xs text-slate-500">18 – 100 years</span>
                </div>
                <input
                  id="age"
                  type="number"
                  min={18}
                  max={100}
                  value={form.age}
                  onChange={(e) => handleChange("age", e.target.value)}
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  required
                />
              </div>

              {/* BMI */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label htmlFor="bmi" className="text-sm font-medium text-slate-700">
                    Body Mass Index (BMI)
                  </label>
                  <span className="text-xs text-slate-500">10.0 – 60.0 kg/m²</span>
                </div>
                <input
                  id="bmi"
                  type="number"
                  step="0.1"
                  min={10}
                  max={60}
                  value={form.bmi}
                  onChange={(e) => handleChange("bmi", e.target.value)}
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  required
                />
                <p className="mt-1 text-xs text-slate-500 flex items-center gap-1">
                  <HelpCircle className="w-3 h-3 text-slate-400" />
                  Standard healthy BMI reference is 18.5 – 24.9.
                </p>
              </div>

              {/* Children */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label htmlFor="children" className="text-sm font-medium text-slate-700">
                    Dependents / Children
                  </label>
                  <span className="text-xs text-slate-500">0 – 20 dependents</span>
                </div>
                <input
                  id="children"
                  type="number"
                  min={0}
                  max={20}
                  value={form.children}
                  onChange={(e) => handleChange("children", e.target.value)}
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  required
                />
              </div>

              {/* Smoker Status */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Tobacco Smoking Status
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => handleChange("smoker", "no")}
                    className={`px-4 py-2.5 text-sm font-medium rounded-lg border transition-all ${
                      form.smoker === "no"
                        ? "bg-blue-50 border-blue-600 text-blue-700 ring-1 ring-blue-600 shadow-sm"
                        : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    Non-Smoker
                  </button>
                  <button
                    type="button"
                    onClick={() => handleChange("smoker", "yes")}
                    className={`px-4 py-2.5 text-sm font-medium rounded-lg border transition-all ${
                      form.smoker === "yes"
                        ? "bg-amber-50 border-amber-600 text-amber-800 ring-1 ring-amber-600 shadow-sm"
                        : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    Smoker
                  </button>
                </div>
              </div>

              {/* Region */}
              <div>
                <label htmlFor="region" className="block text-sm font-medium text-slate-700 mb-1.5">
                  Geographic Region
                </label>
                <select
                  id="region"
                  value={form.region}
                  onChange={(e) => handleChange("region", e.target.value)}
                  className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-slate-900 text-sm focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                >
                  <option value="northeast">Northeast</option>
                  <option value="northwest">Northwest</option>
                  <option value="southeast">Southeast</option>
                  <option value="southwest">Southwest</option>
                </select>
              </div>

              {/* Action Buttons */}
              <div className="pt-2 flex items-center gap-3">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 inline-flex justify-center items-center gap-2 px-5 py-3 rounded-lg text-white bg-blue-600 hover:bg-blue-700 font-semibold text-sm shadow-sm transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Calculating Quote...
                    </>
                  ) : (
                    "Calculate Medical Cost"
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={loading}
                  title="Reset form"
                  className="p-3 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-100 transition-colors disabled:opacity-50"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>

              <div className="text-xs text-slate-400 text-center pt-1">
                Per ACA § 2701 compliance, biological sex is excluded from rating.
              </div>
            </form>
          </div>

          {/* Result Column */}
          <div className="lg:col-span-6 space-y-6">
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-2xl p-5 text-red-800 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-sm font-semibold">Prediction Error</h4>
                  <p className="text-sm mt-1 text-red-700 leading-relaxed">{error}</p>
                </div>
              </div>
            )}

            {!result && !error && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-8 text-center text-slate-500">
                <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4 text-slate-400">
                  <Activity className="w-6 h-6" />
                </div>
                <h3 className="text-base font-semibold text-slate-800">
                  Ready to Predict
                </h3>
                <p className="text-sm text-slate-500 mt-1 max-w-sm mx-auto">
                  Adjust the applicant information and click &quot;Calculate Medical Cost&quot; to view actuarial estimates and risk assessment.
                </p>
              </div>
            )}

            {result && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
                {/* Header status */}
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Underwriting Evaluation
                    </span>
                    <h3 className="text-lg font-bold text-slate-900">
                      Predicted Medical Expenses
                    </h3>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    {result.model_version}
                  </span>
                </div>

                {/* Primary Cost Metric */}
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 text-center">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Expected Annual Cost
                  </span>
                  <div className="text-3xl sm:text-4xl font-extrabold text-slate-900 mt-1">
                    {formatCurrency(result.predicted_cost)}
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    LightGBM point estimate (regression_l1 median)
                  </p>
                </div>

                {/* Secondary Cards: Interval & Tier */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Prediction Interval */}
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 block mb-1">
                      80% Prediction Interval
                    </span>
                    <div className="text-lg font-bold text-slate-900">
                      {formatCurrency(result.interval_80[0])} –{" "}
                      {formatCurrency(result.interval_80[1])}
                    </div>
                    <p className="text-xs text-slate-500 mt-1">
                      Calibrated 10th to 90th quantile interval
                    </p>
                  </div>

                  {/* Underwriting Tier */}
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 block mb-1">
                      Underwriting Risk Tier
                    </span>
                    <div className="mt-1">
                      {result.premium_tier === "Standard" && (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Standard Rate
                        </span>
                      )}
                      {result.premium_tier === "Loaded" && (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">
                          <AlertCircle className="w-3.5 h-3.5" />
                          Loaded (Risk Surcharge)
                        </span>
                      )}
                      {result.premium_tier === "Refer to Underwriter" && (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200">
                          <ShieldAlert className="w-3.5 h-3.5" />
                          Refer to Underwriter
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 mt-2">
                      {result.premium_tier === "Standard" && "Automated policy issuance eligible."}
                      {result.premium_tier === "Loaded" && "Moderate risk profile surcharge."}
                      {result.premium_tier === "Refer to Underwriter" && "Requires specialist manual review."}
                    </p>
                  </div>
                </div>

                {/* Key Cost Drivers (SHAP Reason Codes) */}
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
                    Key Influencing Factors (SHAP Drivers)
                  </h4>
                  {result.reason_codes && result.reason_codes.length > 0 ? (
                    <div className="space-y-2">
                      {result.reason_codes.map((rc, idx) => {
                        const isIncrease = rc.direction === "increases cost";
                        return (
                          <div
                            key={idx}
                            className={`flex items-center justify-between p-3 rounded-lg border text-sm ${
                              isIncrease
                                ? "bg-rose-50/70 border-rose-200 text-rose-900"
                                : "bg-emerald-50/70 border-emerald-200 text-emerald-900"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              {isIncrease ? (
                                <ArrowUpRight className="w-4 h-4 text-rose-600 shrink-0" />
                              ) : (
                                <ArrowDownRight className="w-4 h-4 text-emerald-600 shrink-0" />
                              )}
                              <span className="font-medium capitalize">
                                {rc.feature}
                              </span>
                            </div>
                            <div className="font-semibold text-right">
                              {isIncrease ? "+" : "-"}
                              {formatCurrency(rc.impact)}
                              <span className="block text-[11px] font-normal text-slate-500 capitalize">
                                {rc.direction}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 bg-slate-50 p-3 rounded-lg border border-slate-200">
                      All inputs fall within baseline normative ranges with no dominant outlier driver.
                    </p>
                  )}
                </div>

                {/* Footer action */}
                <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                  <span>Audit timestamp: {new Date(result.decision_timestamp).toLocaleTimeString()}</span>
                  <button
                    onClick={handleReset}
                    className="text-blue-600 hover:text-blue-800 font-medium transition-colors"
                  >
                    Calculate New Prediction
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
