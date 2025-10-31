# Neutreeno ML Engineer Case Study

## Overview
This is the take-home **Machine Learning Engineer Technical Case Study** for **Neutreeno**, an intelligent decarbonisation platform.

You will:
- Work with **unstructured emissions / procurement data**
- Use **out-of-the-box ML tools**
- Deliver a **practical, startup-friendly** solution
- Show how you communicate your work

## Context
Neutreeno is building an **emission factor matching system** that links unstructured company data to a proprietary emissions database.

Your task is to build a **proof-of-concept (POC) pipeline** that maps company procurement items to the **UK DESNZ emission factors**.

---

## Objective
Build an **intelligent system** to map purchased items from company procurement data to **UK DESNZ emission factors** using ML techniques.

---

## Provided Resources

### 1. UK DESNZ Emission Factors Database (Excel)
- 3,000+ emission factors
- Scope 1, 2, 3 (kgCO₂e / unit)
- Category hierarchies & descriptions
- SIC codes and activity descriptions

### 2. Sample Company Purchase Report (Excel)
- 6-month procurement data from fictional **“TechCorp Ltd”**
- 500+ line items
- Product/service descriptions (3 levels, messy naming)
- Quantities and units (mixed)
- Supplier names and categories
- Purchase amounts (GBP)
- **Intentionally messy**: inconsistent naming, abbreviations, typos

---

## Requirements

### 1. Intelligent Matching System
- Create a matching pipeline between purchase descriptions and DESNZ categories
- Use **pre-trained semantic models** (e.g. Sentence-BERT, Hugging Face, similar)
- Handle:
  - different naming conventions
  - typos and abbreviations
  - “Company laptops” → “Computers and IT equipment”
  - different taxonomies

### 2. Uncertainty Quantification
- Assign a **confidence/uncertainty score** to each match
- **Flag ambiguous items** for human review
- Provide **top-3 candidate matches** when confidence is low
- Handle **no suitable emission factor** cases

---

## Evaluation Criteria

- **Smart use of ML models** for semantic matching
- **Robustness** to real-world, messy data
- **Uncertainty handling** and user feedback
- **Code efficiency & scalability** for larger datasets
- **Clarity of communication** on approach and results

---

## Important Notes

1. **Use of AI tools**  
   You may use GitHub Copilot, ChatGPT, etc.  
   You **must document** where you used them and **be able to explain** your code.

2. **Focus on presentation**  
   We mainly want to **verify results** you present.  
   You **won’t be judged** on pristine code quality.

3. **External libraries**  
   Prefer **well-maintained**, established libraries.  
   Document any **trade-offs** (custom vs out-of-the-box).

4. **Time-boxing**  
   Aim for a **working MVP** over a perfect solution.

6. **What we care about most**  
   - Fast evaluation & implementation of out-of-the-box solutions  
   - Pragmatic, startup-style decision-making  
   - Understanding ML **uncertainty** & business impact  
   - Clear communication to different audiences

---
