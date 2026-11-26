# Article-Specific Questions for RAG Evaluation

These ten questions were drafted from facts in the BBC News Articles collection to replace broad prompts with source-grounded retrieval queries. The expected fact is included to make manual relevance checking possible. All ten were tested with BM25, dense FAISS, and hybrid retrieval, then answered with Mistral using Cross-Encoder-reranked hybrid results. The retrieved article, ranked chunks, and generated answers are saved in `outputs/article_question_*.csv`.

For these ten questions, each retrieval method included the expected source article in its top five (10/10 per method). This is a small, source-targeted relevance check, not a general benchmark. Mistral answered most questions accurately but omitted details in some responses; see the answer CSV and report for examples and limitations.

| # | Question | Source article | Expected evidence to retrieve |
|---:|---|---|---|
| 1 | How did TimeWarner's quarterly profit compare with the previous year's figure, and what was the latest amount? | `BBC News Summary/News Articles/business/001.txt` | Quarterly profit was reported as $1.13bn, compared with $639m a year earlier. |
| 2 | What pre-tax profit did WPP report, and by what percentage had it increased? | `BBC News Summary/News Articles/business/035.txt` | Pre-tax profit reached £546m, a 15% increase. |
| 3 | By what average percentage did Marks & Spencer reduce prices in London and the regions? | `BBC News Summary/News Articles/business/184.txt` | The reported average price reduction was 24%. |
| 4 | What actions might the UK government take to avoid breaking its "golden rule"? | `BBC News Summary/News Articles/business/195.txt` | The report says taxes may need to rise or spending be restrained; one estimate put the tax increase near £10bn. |
| 5 | By how much did the IEA revise its 2005 oil-demand growth estimate, and what level of demand did it project? | `BBC News Summary/News Articles/business/028.txt` | The estimate rose by 80,000 barrels per day to 84 million barrels per day. |
| 6 | What annual output was planned for Hyundai's second Indian plant, and what total capacity would that bring? | `BBC News Summary/News Articles/business/054.txt` | The new plant was to produce 150,000 cars a year, taking capacity to 400,000. |
| 7 | What happened to the Ifo business-confidence index between January and February? | `BBC News Summary/News Articles/business/066.txt` | It fell from 97.5 in January to 95.5 in February. |
| 8 | How much of Colombia's subsidy funding was reported to benefit the richest part of the population? | `BBC News Summary/News Articles/business/033.txt` | 12.1 trillion pesos, half of the 24.2-trillion-peso fund, was going to the richest group. |
| 9 | What were Cadbury Schweppes' underlying pre-tax profits in 2004, and how did currency movements affect the reported increase? | `BBC News Summary/News Articles/business/143.txt` | Underlying pre-tax profit was £933m; growth was 1%, or 8% excluding currency movements. |
| 10 | Which two companies agreed to collaborate on flat-screen television technology? | `BBC News Summary/News Articles/business/211.txt` | Hitachi and Matsushita Electric agreed to collaborate. |

The evaluation CSVs preserve category-relative paths such as `business/001.txt` so that filenames remain unambiguous when different categories reuse a basename.
