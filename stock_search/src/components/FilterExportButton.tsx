import React, { useState } from "react";
import type { SearchFilters, SortConfig } from "../types/stock";
import { saveFiltersYaml } from "../utils/filterExport";

interface FilterExportButtonProps {
  filters: SearchFilters;
  sortConfig?: SortConfig | null;
  baseFileName?: string;
  className?: string;
}

export const FilterExportButton: React.FC<FilterExportButtonProps> = ({
  filters,
  sortConfig = null,
  baseFileName = "stock_search",
  className = "",
}) => {
  const [isExporting, setIsExporting] = useState(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  const handleExport = async () => {
    if (isExporting) return;

    setIsExporting(true);
    setExportMessage(null);

    try {
      const method = await saveFiltersYaml(filters, sortConfig, baseFileName);
      setExportMessage(
        method === "file-picker"
          ? "✅ 検索条件を保存しました"
          : "✅ 検索条件をダウンロードしました",
      );
      setTimeout(() => setExportMessage(null), 4000);
    } catch (error) {
      const errorName =
        error instanceof DOMException
          ? error.name
          : (error as { name?: string }).name;

      if (errorName === "AbortError") {
        setExportMessage("🟡 保存をキャンセルしました");
        setTimeout(() => setExportMessage(null), 2500);
      } else {
        console.error("Filter export error:", error);
        setExportMessage("❗ エクスポートに失敗しました");
        setTimeout(() => setExportMessage(null), 4000);
      }
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={handleExport}
        disabled={isExporting}
        className={`btn btn-outline btn-sm gap-2 ${isExporting ? "btn-disabled opacity-70" : ""}`}
      >
        {isExporting ? (
          <>
            <span className="loading loading-spinner loading-xs"></span>
            保存中...
          </>
        ) : (
          <>🧾 YAMLエクスポート</>
        )}
      </button>

      {exportMessage && (
        <div className="absolute top-full left-0 mt-2 px-3 py-2 bg-base-100 border border-base-300 text-sm rounded-lg shadow-lg whitespace-nowrap z-20 min-w-max">
          {exportMessage}
        </div>
      )}
    </div>
  );
};
