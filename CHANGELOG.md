# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Hidden watermark detector engine (21st engine)
  - Detects zero-width characters, invisible characters, and AI watermarks
  - Supports 20+ types of hidden characters
- English and Chinese AI language markers expansion
- Chinese transition words support

### Changed
- Engine weights adjusted for better accuracy
- All 21 engines have been polished and refined
- n-gram perplexity algorithm improved
- Sliding window detection window size increased from 5 to 7
- Burstiness thresholds adjusted
- Statistical baseline parameters optimized
- Compressibility algorithm refined
- Structured shape detection enhanced with table detection

### Polished Engines
1. **human-imperfection** - Expanded typo patterns, added punctuation patterns
2. **emotion-curve** - Improved emotional word lists, adjusted thresholds
3. **specific-details** - Expanded number patterns, added date patterns
4. **self-references** - Expanded first-person phrases, added patterns
5. **logical-connectors** - Expanded connector lists, added patterns
6. **formulaic-patterns** - Expanded opener lists, added patterns
7. **linguistic-markers** - Expanded marker lists, added Chinese support
8. **vocabulary-richness** - Added short word ratio metric
9. **structural-patterns** - Added semicolon and dash frequency metrics
10. **sentiment-hedging** - Expanded hedging phrases, expanded vague positivity lists
11. **formulaic-markers** - Expanded marker lists significantly
12. **transition-density** - Expanded transition words list
13. **compressibility** - Adjusted normalization parameters
14. **structured-shape** - Added table detection
15. **hidden-characters** - Expanded hidden character list to 20+ types
16. **ngram-perplexity** - Adjusted normalization parameters
17. **sliding-window** - Increased window size, adjusted thresholds
18. **burstiness-stat** - Adjusted thresholds
19. **statistical-baseline** - Adjusted thresholds

## [1.0.0] - 2026-10-26

### Added
- Initial release with 20 detection engines
- RoBERTa ML classifier integration
- 5 novel innovation engines
- HTML report generation
- Local document comparison
- Automatic citation exclusion
