//! Enhanced metrics export capabilities
//!
//! Provides multiple export formats:
//! - Prometheus metrics format
//! - JSON export
//! - CSV export
//! - Console pretty printing

use std::collections::HashMap;

use crate::core::error::Result;
use super::metrics::MetricsReport;

/// Metrics exporter trait
pub trait MetricsExporter: Send + Sync {
    /// Export metrics to string
    fn export(&self, report: &MetricsReport) -> Result<String>;
    
    /// Export format name
    fn format_name(&self) -> &'static str;
}

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    /// Metrics prefix
    prefix: String,
    /// Include timestamp
    include_timestamp: bool,
}

impl PrometheusExporter {
    /// Create new Prometheus exporter
    pub fn new() -> Self {
        Self {
            prefix: "tts".to_string(),
            include_timestamp: true,
        }
    }
    
    /// Set metrics prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }
    
    /// Enable/disable timestamp
    pub fn with_timestamp(mut self, enable: bool) -> Self {
        self.include_timestamp = enable;
        self
    }
    
    /// Format timer as Prometheus metric
    fn format_timer(&self, timer: &super::metrics::TimerReport) -> String {
        let name = format!("{}_{}_duration", self.prefix, timer.name.replace('.', "_"));
        
        let mut output = String::new();
        
        // Help
        output.push_str(&format!(
            "# HELP {} Duration of {} operation in milliseconds\n",
            name, timer.name
        ));
        
        // Type
        output.push_str(&format!("# TYPE {} summary\n", name));
        
        // Count
        output.push_str(&format!(
            "{}_count {}\n",
            name, timer.count
        ));
        
        // Sum
        output.push_str(&format!(
            "{}_sum {:.2}\n",
            name, timer.avg_ms * timer.count as f64
        ));
        
        // Average
        output.push_str(&format!(
            "{}{{quantile=\"0.5\"}} {:.2}\n",
            name, timer.avg_ms
        ));
        
        if self.include_timestamp {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis();
            output.push_str(&format!("# Timestamp: {}\n", timestamp));
        }
        
        output
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsExporter for PrometheusExporter {
    fn export(&self, report: &MetricsReport) -> Result<String> {
        let mut output = String::new();
        
        // Export timers
        for timer in &report.timers {
            output.push_str(&self.format_timer(timer));
            output.push('\n');
        }
        
        // Export counters
        for (name, value) in &report.counters {
            let metric_name = format!("{}_{}", self.prefix, name.replace('.', "_"));
            output.push_str(&format!("# HELP {} Counter\n", metric_name));
            output.push_str(&format!("# TYPE {} counter\n", metric_name));
            output.push_str(&format!("{} {}\n", metric_name, value));
        }
        
        if !report.counters.is_empty() {
            output.push('\n');
        }
        
        // Export gauges
        for (name, value) in &report.gauges {
            let metric_name = format!("{}_{}", self.prefix, name.replace('.', "_"));
            output.push_str(&format!("# HELP {} Gauge\n", metric_name));
            output.push_str(&format!("# TYPE {} gauge\n", metric_name));
            output.push_str(&format!("{} {:.6}\n", metric_name, value));
        }
        
        Ok(output)
    }
    
    fn format_name(&self) -> &'static str {
        "prometheus"
    }
}

/// JSON metrics exporter
pub struct JsonExporter {
    /// Pretty print JSON
    pretty: bool,
    /// Include metadata
    include_metadata: bool,
}

impl JsonExporter {
    /// Create new JSON exporter
    pub fn new() -> Self {
        Self {
            pretty: true,
            include_metadata: true,
        }
    }
    
    /// Enable/disable pretty print
    pub fn pretty(mut self, enable: bool) -> Self {
        self.pretty = enable;
        self
    }
    
    /// Enable/disable metadata
    pub fn with_metadata(mut self, enable: bool) -> Self {
        self.include_metadata = enable;
        self
    }
}

impl Default for JsonExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsExporter for JsonExporter {
    fn export(&self, report: &MetricsReport) -> Result<String> {
        use serde_json::json;
        
        let mut timers = Vec::new();
        for timer in &report.timers {
            timers.push(json!({
                "name": timer.name,
                "count": timer.count,
                "avg_ms": timer.avg_ms,
                "min_ms": timer.min_ms,
                "max_ms": timer.max_ms,
            }));
        }
        
        let mut counters = HashMap::new();
        for (name, value) in &report.counters {
            counters.insert(name.clone(), value);
        }
        
        let mut gauges = HashMap::new();
        for (name, value) in &report.gauges {
            gauges.insert(name.clone(), value);
        }
        
        let mut root = json!({
            "timers": timers,
            "counters": counters,
            "gauges": gauges,
        });
        
        if self.include_metadata {
            root["metadata"] = json!({
                "exported_at": chrono_lite_timestamp(),
                "format_version": "1.0",
            });
        }
        
        let output = if self.pretty {
            serde_json::to_string_pretty(&root).map_err(|e| {
                crate::core::error::TtsError::Internal {
                    message: format!("JSON serialization failed: {}", e),
                    location: None,
                }
            })?
        } else {
            serde_json::to_string(&root).map_err(|e| {
                crate::core::error::TtsError::Internal {
                    message: format!("JSON serialization failed: {}", e),
                    location: None,
                }
            })?
        };
        
        Ok(output)
    }
    
    fn format_name(&self) -> &'static str {
        "json"
    }
}

/// CSV metrics exporter
pub struct CsvExporter {
    /// Include header row
    include_header: bool,
}

impl CsvExporter {
    /// Create new CSV exporter
    pub fn new() -> Self {
        Self {
            include_header: true,
        }
    }
    
    /// Enable/disable header
    pub fn with_header(mut self, enable: bool) -> Self {
        self.include_header = enable;
        self
    }
}

impl Default for CsvExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsExporter for CsvExporter {
    fn export(&self, report: &MetricsReport) -> Result<String> {
        let mut output = String::new();
        
        if self.include_header {
            output.push_str("type,name,count,avg_ms,min_ms,max_ms,value\n");
        }
        
        // Export timers
        for timer in &report.timers {
            output.push_str(&format!(
                "timer,{}, {},{:.2},{:.2},{:.2},\n",
                timer.name, timer.count, timer.avg_ms, timer.min_ms, timer.max_ms
            ));
        }
        
        // Export counters
        for (name, value) in &report.counters {
            output.push_str(&format!(
                "counter,{},,{},,,{}\n",
                name, value, value
            ));
        }
        
        // Export gauges
        for (name, value) in &report.gauges {
            output.push_str(&format!(
                "gauge,{},,,,,{:.6}\n",
                name, value
            ));
        }
        
        Ok(output)
    }
    
    fn format_name(&self) -> &'static str {
        "csv"
    }
}

/// Console pretty printer
pub struct ConsoleExporter {
    /// Use colors
    use_colors: bool,
    /// Include empty sections
    include_empty: bool,
}

impl ConsoleExporter {
    /// Create new console exporter
    pub fn new() -> Self {
        Self {
            use_colors: true,
            include_empty: false,
        }
    }
    
    /// Enable/disable colors
    pub fn colored(mut self, enable: bool) -> Self {
        self.use_colors = enable;
        self
    }
    
    /// Enable/disable empty sections
    pub fn with_empty_sections(mut self, enable: bool) -> Self {
        self.include_empty = enable;
        self
    }
}

impl Default for ConsoleExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsoleExporter {
    fn format_section(&self, title: &str, content: &str) -> String {
        let width = 60;
        let mut output = String::new();
        
        if self.use_colors {
            output.push_str("\x1b[1m\x1b[36m"); // Bold cyan
        }
        
        output.push_str(&"=".repeat(width));
        output.push('\n');
        output.push_str(title);
        output.push('\n');
        output.push_str(&"=".repeat(width));
        
        if self.use_colors {
            output.push_str("\x1b[0m"); // Reset
        }
        
        output.push('\n');
        output.push_str(content);
        output.push('\n');
        
        output
    }
    
    fn format_timer(&self, timer: &super::metrics::TimerReport) -> String {
        let mut output = String::new();
        
        if self.use_colors {
            output.push_str(&format!(
                "  \x1b[1m{}\x1b[0m: {} calls, avg=\x1b[32m{:.2}ms\x1b[0m, min={:.2}ms, max={:.2}ms\n",
                timer.name, timer.count, timer.avg_ms, timer.min_ms, timer.max_ms
            ));
        } else {
            output.push_str(&format!(
                "  {}: {} calls, avg={:.2}ms, min={:.2}ms, max={:.2}ms\n",
                timer.name, timer.count, timer.avg_ms, timer.min_ms, timer.max_ms
            ));
        }
        
        output
    }
}

impl MetricsExporter for ConsoleExporter {
    fn export(&self, report: &MetricsReport) -> Result<String> {
        let mut output = String::new();
        
        // Timers section
        if !report.timers.is_empty() || self.include_empty {
            let mut timers_content = String::new();
            for timer in &report.timers {
                timers_content.push_str(&self.format_timer(timer));
            }
            
            if !timers_content.is_empty() {
                output.push_str(&self.format_section("=== Timers ===", &timers_content));
            }
        }
        
        // Counters section
        if !report.counters.is_empty() || self.include_empty {
            let mut counters_content = String::new();
            for (name, value) in &report.counters {
                if self.use_colors {
                    counters_content.push_str(&format!(
                        "  \x1b[1m{}\x1b[0m: \x1b[33m{}\x1b[0m\n",
                        name, value
                    ));
                } else {
                    counters_content.push_str(&format!("  {}: {}\n", name, value));
                }
            }
            
            if !counters_content.is_empty() {
                output.push_str(&self.format_section("=== Counters ===", &counters_content));
            }
        }
        
        // Gauges section
        if !report.gauges.is_empty() || self.include_empty {
            let mut gauges_content = String::new();
            for (name, value) in &report.gauges {
                if self.use_colors {
                    gauges_content.push_str(&format!(
                        "  \x1b[1m{}\x1b[0m: \x1b[34m{:.2}\x1b[0m\n",
                        name, value
                    ));
                } else {
                    gauges_content.push_str(&format!("  {}: {:.2}\n", name, value));
                }
            }
            
            if !gauges_content.is_empty() {
                output.push_str(&self.format_section("=== Gauges ===", &gauges_content));
            }
        }
        
        Ok(output)
    }
    
    fn format_name(&self) -> &'static str {
        "console"
    }
}

/// Multi-format exporter
pub struct MultiExporter {
    exporters: HashMap<String, Box<dyn MetricsExporter>>,
}

impl MultiExporter {
    /// Create new multi exporter
    pub fn new() -> Self {
        Self {
            exporters: HashMap::new(),
        }
    }
    
    /// Add an exporter
    pub fn add_exporter<E: MetricsExporter + 'static>(&mut self, exporter: E) {
        self.exporters.insert(exporter.format_name().to_string(), Box::new(exporter));
    }
    
    /// Export to all formats
    pub fn export_all(&self, report: &MetricsReport) -> Result<HashMap<String, String>> {
        let mut results = HashMap::new();
        
        for (name, exporter) in &self.exporters {
            results.insert(name.clone(), exporter.export(report)?);
        }
        
        Ok(results)
    }
    
    /// Export to specific format
    pub fn export_format(&self, format: &str, report: &MetricsReport) -> Result<String> {
        self.exporters.get(format)
            .ok_or_else(|| crate::core::error::TtsError::Internal {
                message: format!("Exporter format '{}' not found", format),
                location: None,
            })?
            .export(report)
    }
}

impl Default for MultiExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple timestamp function (avoiding chrono dependency)
fn chrono_lite_timestamp() -> String {
    use std::time::SystemTime;
    
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    
    let secs = duration.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

/// Unix epoch for timestamp calculations
static UNIX_EPOCH: std::time::SystemTime = std::time::SystemTime::UNIX_EPOCH;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::metrics::{TimerReport, MetricsReport};
    
    fn create_test_report() -> MetricsReport {
        let mut report = MetricsReport::default();
        
        report.timers.push(TimerReport {
            name: "inference".to_string(),
            count: 100,
            avg_ms: 250.5,
            min_ms: 100.0,
            max_ms: 500.0,
        });
        
        report.counters.push(("requests".to_string(), 1000));
        report.gauges.push(("memory_mb".to_string(), 512.5));
        
        report
    }
    
    #[test]
    fn test_prometheus_exporter() {
        let exporter = PrometheusExporter::new();
        let report = create_test_report();
        
        let output = exporter.export(&report).unwrap();
        
        assert!(output.contains("tts_inference_duration"));
        assert!(output.contains("HELP"));
        assert!(output.contains("TYPE"));
    }
    
    #[test]
    fn test_json_exporter() {
        let exporter = JsonExporter::new();
        let report = create_test_report();
        
        let output = exporter.export(&report).unwrap();
        
        assert!(output.contains("\"timers\""));
        assert!(output.contains("\"inference\""));
        assert!(output.contains("\"counters\""));
    }
    
    #[test]
    fn test_csv_exporter() {
        let exporter = CsvExporter::new();
        let report = create_test_report();
        
        let output = exporter.export(&report).unwrap();
        
        assert!(output.contains("type,name,count"));
        assert!(output.contains("timer,inference"));
    }
    
    #[test]
    fn test_multi_exporter() {
        let mut multi = MultiExporter::new();
        multi.add_exporter(PrometheusExporter::new());
        multi.add_exporter(JsonExporter::new());
        
        let report = create_test_report();
        let results = multi.export_all(&report).unwrap();
        
        assert!(results.contains_key("prometheus"));
        assert!(results.contains_key("json"));
    }
}
