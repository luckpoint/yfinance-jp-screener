import type { SearchFilters, SortConfig } from "../types/stock";
import { DATE_FORMAT } from "../constants/formatting";

type FileSystemWritableFileStreamLike = {
  write: (data: Blob) => Promise<void>;
  close: () => Promise<void>;
};

type FileSystemFileHandleLike = {
  createWritable: () => Promise<FileSystemWritableFileStreamLike>;
};

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: Array<{
    description?: string;
    accept: Record<string, string[]>;
  }>;
};

type SaveFilePicker = (
  options?: SaveFilePickerOptions,
) => Promise<FileSystemFileHandleLike>;

const YAML_MIME_TYPE = "application/x-yaml";
const YAML_EXTENSIONS = [".yaml", ".yml"];

const getSaveFilePicker = (): SaveFilePicker | undefined => {
  if (typeof window === "undefined") return undefined;

  const picker = (
    window as Window & {
      showSaveFilePicker?: SaveFilePicker;
    }
  ).showSaveFilePicker;

  return typeof picker === "function" ? picker : undefined;
};

const downloadTextFile = (
  content: string,
  filename: string,
  mimeType: string,
): void => {
  const blob = new Blob([content], {
    type: `${mimeType};charset=utf-8`,
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

const escapeYamlString = (value: string): string => {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
};

const formatYamlValue = (value: string | number | boolean): string => {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toString() : "null";
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  return `"${escapeYamlString(value)}"`;
};

const normalizeFiltersForExport = (
  filters: SearchFilters,
): Array<[keyof SearchFilters, string | number | string[]]> => {
  const entries = Object.entries(filters) as Array<
    [keyof SearchFilters, SearchFilters[keyof SearchFilters]]
  >;

  return entries.reduce<Array<[keyof SearchFilters, string | number | string[]]>>(
    (acc, [key, value]) => {
      if (Array.isArray(value)) {
        const cleaned = value.filter((item) => item.trim().length > 0);
        if (cleaned.length > 0) {
          acc.push([key, cleaned]);
        }
        return acc;
      }

      if (typeof value === "string") {
        if (value.trim().length > 0) {
          acc.push([key, value]);
        }
        return acc;
      }

      if (typeof value === "number") {
        if (Number.isFinite(value)) {
          acc.push([key, value]);
        }
        return acc;
      }

      return acc;
    },
    [],
  );
};

export const serializeFiltersToYaml = (
  filters: SearchFilters,
  sortConfig?: SortConfig | null,
): string => {
  const entries = normalizeFiltersForExport(filters);
  const lines: string[] = [];

  if (entries.length === 0) {
    lines.push("filters: {}");
  } else {
    lines.push("filters:");
    entries.forEach(([key, value]) => {
      if (Array.isArray(value)) {
        lines.push(`  ${key}:`);
        value.forEach((item) => {
          lines.push(`    - ${formatYamlValue(item)}`);
        });
      } else {
        lines.push(`  ${key}: ${formatYamlValue(value)}`);
      }
    });
  }

  if (sortConfig && String(sortConfig.key).trim().length > 0) {
    lines.push("sort:");
    lines.push(`  key: ${formatYamlValue(String(sortConfig.key))}`);
    lines.push(`  direction: ${formatYamlValue(sortConfig.direction)}`);
  }

  return `${lines.join("\n")}\n`;
};

export const generateFilterExportFileName = (
  baseFileName: string = "stock_search",
): string => {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(
    DATE_FORMAT.zeroPadLength,
    DATE_FORMAT.zeroPadChar,
  );
  const day = String(now.getDate()).padStart(
    DATE_FORMAT.zeroPadLength,
    DATE_FORMAT.zeroPadChar,
  );
  const hours = String(now.getHours()).padStart(
    DATE_FORMAT.zeroPadLength,
    DATE_FORMAT.zeroPadChar,
  );
  const minutes = String(now.getMinutes()).padStart(
    DATE_FORMAT.zeroPadLength,
    DATE_FORMAT.zeroPadChar,
  );
  const safeBaseName = baseFileName.trim() || "stock_search";

  return `${safeBaseName}_filters_${year}${month}${day}_${hours}${minutes}.yaml`;
};

export const saveFiltersYaml = async (
  filters: SearchFilters,
  sortConfig?: SortConfig | null,
  baseFileName?: string,
): Promise<"file-picker" | "download"> => {
  const yaml = serializeFiltersToYaml(filters, sortConfig);
  const fileName = generateFilterExportFileName(baseFileName);
  const picker = getSaveFilePicker();

  if (picker) {
    const handle = await picker({
      suggestedName: fileName,
      types: [
        {
          description: "YAML",
          accept: {
            [YAML_MIME_TYPE]: YAML_EXTENSIONS,
          },
        },
      ],
    });
    const writable = await handle.createWritable();
    await writable.write(new Blob([yaml], { type: YAML_MIME_TYPE }));
    await writable.close();
    return "file-picker";
  }

  downloadTextFile(yaml, fileName, YAML_MIME_TYPE);
  return "download";
};
