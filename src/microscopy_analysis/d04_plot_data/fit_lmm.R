#!/usr/bin/env Rscript
# fit_lmm.R
#
# Fit a linear mixed model with lme4 (REML), Satterthwaite-corrected
# Type-III ANOVA via lmerTest, and per-timepoint pairwise group
# contrasts via emmeans. Writes a JSON results bundle for consumption
# by the Python wrapper run_stats_lmm_r.run_lmm_stats.
#
# Designed to be called as a subprocess from Python — no R-side state
# carries over between invocations.
#
# Usage:
#   Rscript fit_lmm.R \
#     --csv data.csv \
#     --response y \
#     --time time \
#     --group group \
#     --subject experiment \
#     [--cell UID] \
#     --output results.json
#
# Required R packages: lme4, lmerTest, emmeans, jsonlite.
# Install once with:
#   install.packages(c('lme4', 'lmerTest', 'emmeans', 'jsonlite'))

suppressPackageStartupMessages({
  ok <- TRUE
  for (pkg in c('lme4', 'lmerTest', 'emmeans', 'jsonlite')) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("ERROR: required R package not installed: %s\n", pkg),
          file = stderr())
      ok <- FALSE
    }
  }
  if (!ok) {
    cat("Install with: install.packages(c('lme4','lmerTest','emmeans','jsonlite'))\n",
        file = stderr())
    quit(status = 2)
  }
  library(lmerTest)
  library(emmeans)
  library(jsonlite)
})

# --- Argument parsing ------------------------------------------------------

parse_args <- function(args) {
  if (length(args) %% 2 != 0) {
    stop("Arguments must be supplied in --key value pairs.")
  }
  out <- list()
  for (i in seq(1, length(args), by = 2)) {
    key <- sub("^--", "", args[i])
    out[[key]] <- args[i + 1]
  }
  out
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
required <- c("csv", "response", "time", "group", "subject", "output")
missing <- setdiff(required, names(opts))
if (length(missing) > 0) {
  cat(sprintf("ERROR: missing required arguments: %s\n",
              paste(missing, collapse = ", ")), file = stderr())
  quit(status = 2)
}
cell_col <- if (!is.null(opts$cell) && nchar(opts$cell) > 0) opts$cell else NULL

# --- Load data -------------------------------------------------------------

df <- read.csv(opts$csv, stringsAsFactors = FALSE, check.names = FALSE)
needed_cols <- c(opts$response, opts$time, opts$group, opts$subject, cell_col)
needed_cols <- needed_cols[!is.null(needed_cols)]
absent <- setdiff(needed_cols, names(df))
if (length(absent) > 0) {
  cat(sprintf("ERROR: input CSV missing columns: %s\n",
              paste(absent, collapse = ", ")), file = stderr())
  quit(status = 3)
}

# Drop rows missing the response.
df <- df[!is.na(df[[opts$response]]), , drop = FALSE]
if (nrow(df) == 0) {
  cat("ERROR: no non-missing rows for the response.\n", file = stderr())
  quit(status = 3)
}

# Cast group / time / subject / cell to factors so the model treats them
# as categorical and emmeans iterates the factor levels.
df[[opts$time]] <- as.factor(df[[opts$time]])
df[[opts$group]] <- as.factor(df[[opts$group]])
df[[opts$subject]] <- as.factor(df[[opts$subject]])
if (!is.null(cell_col)) {
  df[[cell_col]] <- as.factor(df[[cell_col]])
}

# --- Build formula ---------------------------------------------------------

bt <- function(x) sprintf("`%s`", x)  # backtick-quote names with spaces / special chars

fixed <- sprintf("%s ~ %s * %s",
                 bt(opts$response), bt(opts$time), bt(opts$group))
if (!is.null(cell_col)) {
  random <- sprintf("(1 | %s) + (1 | %s:%s)",
                    bt(opts$subject), bt(opts$subject), bt(cell_col))
} else {
  random <- sprintf("(1 | %s)", bt(opts$subject))
}
formula_str <- sprintf("%s + %s", fixed, random)
form <- as.formula(formula_str)

# --- Fit ------------------------------------------------------------------

model <- tryCatch(
  lmer(form, data = df, REML = TRUE),
  error = function(e) {
    cat(sprintf("ERROR: lmer fit failed: %s\n", conditionMessage(e)),
        file = stderr())
    quit(status = 4)
  }
)

# --- Main effects: Type-III ANOVA, Satterthwaite df -----------------------

aov <- tryCatch(
  anova(model, type = "III", ddf = "Satterthwaite"),
  error = function(e) NULL
)

main_effects <- list()
if (!is.null(aov)) {
  for (term in rownames(aov)) {
    main_effects[[length(main_effects) + 1]] <- list(
      term     = term,
      F_value  = unname(aov[term, "F value"]),
      df_num   = unname(aov[term, "NumDF"]),
      df_den   = unname(aov[term, "DenDF"]),
      p_value  = unname(aov[term, "Pr(>F)"])
    )
  }
}

# --- Per-timepoint pairwise contrasts via emmeans -------------------------

posthoc <- list()
emm <- tryCatch(
  emmeans(model, as.formula(sprintf("~ %s | %s", bt(opts$group), bt(opts$time)))),
  error = function(e) NULL
)
if (!is.null(emm)) {
  ctrs <- tryCatch(
    summary(pairs(emm, adjust = "none")),
    error = function(e) NULL
  )
  if (!is.null(ctrs)) {
    time_col <- opts$time
    for (i in seq_len(nrow(ctrs))) {
      posthoc[[length(posthoc) + 1]] <- list(
        timepoint = as.character(ctrs[[time_col]][i]),
        contrast  = as.character(ctrs[["contrast"]][i]),
        estimate  = unname(ctrs[["estimate"]][i]),
        SE        = unname(ctrs[["SE"]][i]),
        df        = unname(ctrs[["df"]][i]),
        t_ratio   = unname(ctrs[["t.ratio"]][i]),
        p_value   = unname(ctrs[["p.value"]][i])
      )
    }
  }
}

# --- Variance components --------------------------------------------------

vc_df <- as.data.frame(VarCorr(model))
variance_components <- list()
for (i in seq_len(nrow(vc_df))) {
  key <- vc_df$grp[i]
  variance_components[[key]] <- unname(vc_df$vcov[i])
}

# --- Convergence diagnostics ----------------------------------------------

conv <- list(
  is_singular = isSingular(model),
  messages    = as.character(model@optinfo$conv$lme4$messages)
)

# --- Build and write output JSON ------------------------------------------

result <- list(
  formula             = formula_str,
  n_obs               = nrow(df),
  n_subjects          = length(levels(df[[opts$subject]])),
  n_cells             = if (!is.null(cell_col)) length(levels(df[[cell_col]])) else NA,
  main_effects        = main_effects,
  posthoc             = posthoc,
  variance_components = variance_components,
  convergence         = conv,
  package_versions    = list(
    R        = paste(R.version$major, R.version$minor, sep = "."),
    lme4     = as.character(packageVersion("lme4")),
    lmerTest = as.character(packageVersion("lmerTest")),
    emmeans  = as.character(packageVersion("emmeans"))
  )
)

write_json(result, opts$output, auto_unbox = TRUE, pretty = TRUE,
           na = "null", null = "null")
