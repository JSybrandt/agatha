from lxml import etree
import io
from pymoliere.construct.parse_pubmed_xml import pubmed_xml_to_record

def test_pubmed_xml_to_record_structured():
  records = []
  with io.BytesIO(bytearray(STRUCTURED_ABSTRACT_XML, encoding="utf-8")) as xml_file:
    records = [
        pubmed_xml_to_record(pm_elem)
        for _, pm_elem in etree.iterparse(xml_file, tag="PubmedArticle")
     ]
  print(records)
  assert len(records) == 1
  actual = records[0]

  assert actual["pmid"] == 31479209
  assert actual["version"] == 1
  assert actual["language"] == "eng"
  assert actual["text_data"][0]["text"].startswith("A Genotype-")
  assert actual["text_data"][0]["type"] == "title"
  assert actual["text_data"][-1]["text"].startswith("In patients undergoing")
  assert actual["text_data"][-1]["type"] == "abstract:conclusions"
  assert actual["publication_types"] == ["Journal Article"]
  assert actual["mesh_headings"] == []
  assert actual["medline_status"] == "Publisher"

def test_pubmed_xml_to_record_unstructured():
  records = []
  with io.BytesIO(bytearray(UNSTRUCTURED_AB_XML, encoding="utf-8")) as xml_file:
    records = [
        pubmed_xml_to_record(pm_elem)
        for _, pm_elem in etree.iterparse(xml_file, tag="PubmedArticle")
     ]
  print(records)
  assert len(records) == 1
  actual = records[0]

  assert actual["pmid"] == 31474367
  assert actual["version"] == 1
  assert actual["language"] == "eng"
  assert actual["text_data"][0]["text"].startswith("Anti-CRISPR-")
  assert actual["text_data"][0]["type"] == "title"
  assert actual["text_data"][-1]["text"].startswith("Phages")
  assert actual["text_data"][-1]["type"] == "abstract:raw"


STRUCTURED_ABSTRACT_XML="""
<PubmedArticle>
    <MedlineCitation Status="Publisher" Owner="NLM">
        <PMID Version="1">31479209</PMID>
        <DateRevised>
            <Year>2019</Year>
            <Month>09</Month>
            <Day>03</Day>
        </DateRevised>
        <Article PubModel="Print-Electronic">
            <Journal>
                <ISSN IssnType="Electronic">1533-4406</ISSN>
                <JournalIssue CitedMedium="Internet">
                    <PubDate>
                        <Year>2019</Year>
                        <Month>Sep</Month>
                        <Day>03</Day>
                    </PubDate>
                </JournalIssue>
                <Title>The New England journal of medicine</Title>
                <ISOAbbreviation>N. Engl. J. Med.</ISOAbbreviation>
            </Journal>
            <ArticleTitle>A Genotype-Guided Strategy for Oral P2Y<sub>12</sub> Inhibitors in Primary PCI.</ArticleTitle>
            <ELocationID EIdType="doi" ValidYN="Y">10.1056/NEJMoa1907096</ELocationID>
            <Abstract>
                <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">It is unknown whether patients undergoing primary percutaneous coronary intervention (PCI) benefit from genotype-guided selection of oral P2Y<sub>12</sub> inhibitors.</AbstractText>
                <AbstractText Label="METHODS" NlmCategory="METHODS">We conducted a randomized, open-label, assessor-blinded trial in which patients undergoing primary PCI with stent implantation were assigned in a 1:1 ratio to receive either a P2Y<sub>12</sub> inhibitor on the basis of early <i>CYP2C19</i> genetic testing (genotype-guided group) or standard treatment with either ticagrelor or prasugrel (standard-treatment group) for 12 months. In the genotype-guided group, carriers of <i>CYP2C19</i>*2 or <i>CYP2C19</i>*3 loss-of-function alleles received ticagrelor or prasugrel, and noncarriers received clopidogrel. The two primary outcomes were net adverse clinical events - defined as death from any cause, myocardial infarction, definite stent thrombosis, stroke, or major bleeding defined according to Platelet Inhibition and Patient Outcomes (PLATO) criteria - at 12 months (primary combined outcome; tested for noninferiority, with a noninferiority margin of 2 percentage points for the absolute difference) and PLATO major or minor bleeding at 12 months (primary bleeding outcome).</AbstractText>
                <AbstractText Label="RESULTS" NlmCategory="RESULTS">For the primary analysis, 2488 patients were included: 1242 in the genotype-guided group and 1246 in the standard-treatment group. The primary combined outcome occurred in 63 patients (5.1%) in the genotype-guided group and in 73 patients (5.9%) in the standard-treatment group (absolute difference, -0.7 percentage points; 95% confidence interval [CI], -2.0 to 0.7; P&lt;0.001 for noninferiority). The primary bleeding outcome occurred in 122 patients (9.8%) in the genotype-guided group and in 156 patients (12.5%) in the standard-treatment group (hazard ratio, 0.78; 95% CI, 0.61 to 0.98; P = 0.04).</AbstractText>
                <AbstractText Label="CONCLUSIONS" NlmCategory="CONCLUSIONS">In patients undergoing primary PCI, a <i>CYP2C19</i> genotype-guided strategy for selection of oral P2Y<sub>12</sub> inhibitor therapy was noninferior to standard treatment with ticagrelor or prasugrel at 12 months with respect to thrombotic events and resulted in a lower incidence of bleeding. (Funded by the Netherlands Organization for Health Research and Development; POPular Genetics ClinicalTrials.gov number, NCT01761786; Netherlands Trial Register number, NL2872.).</AbstractText>
                <CopyrightInformation>Copyright © 2019 Massachusetts Medical Society.</CopyrightInformation>
            </Abstract>
            <AuthorList CompleteYN="Y">
                <Author ValidYN="Y">
                    <LastName>Claassens</LastName>
                    <ForeName>Daniel M F</ForeName>
                    <Initials>DMF</Initials>
                    <Identifier Source="ORCID">0000-0001-6338-2569</Identifier>
                    <AffiliationInfo>
                        <Affiliation>From the Department of Cardiology, St. Antonius Hospital.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Vos</LastName>
                    <ForeName>Gerrit J A</ForeName>
                    <Initials>GJA</Initials>
                    <AffiliationInfo>
                        <Affiliation>
                          From the Department of Cardiology, St. Antonius
                          Hospital.
                        </Affiliation>
                    </AffiliationInfo>
                </Author>
            </AuthorList>
            <Language>eng</Language>
            <DataBankList CompleteYN="Y">
                <DataBank>
                    <DataBankName>ClinicalTrials.gov</DataBankName>
                    <AccessionNumberList>
                        <AccessionNumber>NCT01761786</AccessionNumber>
                    </AccessionNumberList>
                </DataBank>
                <DataBank>
                    <DataBankName>NTR</DataBankName>
                    <AccessionNumberList>
                        <AccessionNumber>NL2872</AccessionNumber>
                    </AccessionNumberList>
                </DataBank>
            </DataBankList>
            <PublicationTypeList>
                <PublicationType UI="D016428">Journal Article</PublicationType>
            </PublicationTypeList>
            <ArticleDate DateType="Electronic">
                <Year>2019</Year>
                <Month>09</Month>
                <Day>03</Day>
            </ArticleDate>
        </Article>
        <MedlineJournalInfo>
            <Country>United States</Country>
            <MedlineTA>N Engl J Med</MedlineTA>
            <NlmUniqueID>0255562</NlmUniqueID>
            <ISSNLinking>0028-4793</ISSNLinking>
        </MedlineJournalInfo>
        <CitationSubset>AIM</CitationSubset>
        <CitationSubset>IM</CitationSubset>
    </MedlineCitation>
    <PubmedData>
        <History>
            <PubMedPubDate PubStatus="entrez">
                <Year>2019</Year>
                <Month>9</Month>
                <Day>4</Day>
                <Hour>6</Hour>
                <Minute>0</Minute>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="pubmed">
                <Year>2019</Year>
                <Month>9</Month>
                <Day>4</Day>
                <Hour>6</Hour>
                <Minute>0</Minute>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="medline">
                <Year>2019</Year>
                <Month>9</Month>
                <Day>4</Day>
                <Hour>6</Hour>
                <Minute>0</Minute>
            </PubMedPubDate>
        </History>
        <PublicationStatus>aheadofprint</PublicationStatus>
        <ArticleIdList>
            <ArticleId IdType="pubmed">31479209</ArticleId>
            <ArticleId IdType="doi">10.1056/NEJMoa1907096</ArticleId>
        </ArticleIdList>
    </PubmedData>
</PubmedArticle>
"""

UNSTRUCTURED_AB_XML = """
<PubmedArticle>
    <MedlineCitation Status="Publisher" Owner="NLM">
        <PMID Version="1">31474367</PMID>
        <DateRevised>
            <Year>2019</Year>
            <Month>09</Month>
            <Day>02</Day>
        </DateRevised>
        <Article PubModel="Print-Electronic">
            <Journal>
                <ISSN IssnType="Electronic">1097-4172</ISSN>
                <JournalIssue CitedMedium="Internet">
                    <PubDate>
                        <Year>2019</Year>
                        <Month>Aug</Month>
                        <Day>27</Day>
                    </PubDate>
                </JournalIssue>
                <Title>Cell</Title>
                <ISOAbbreviation>Cell</ISOAbbreviation>
            </Journal>
            <ArticleTitle>Anti-CRISPR-Associated Proteins Are Crucial Repressors of Anti-CRISPR Transcription.</ArticleTitle>
            <ELocationID EIdType="pii" ValidYN="Y">S0092-8674(19)30846-3</ELocationID>
            <ELocationID EIdType="doi" ValidYN="Y">10.1016/j.cell.2019.07.046</ELocationID>
            <Abstract>
                <AbstractText>Phages express anti-CRISPR (Acr) proteins to inhibit CRISPR-Cas systems that would otherwise destroy their genomes. Most acr genes are located adjacent to anti-CRISPR-associated (aca) genes, which encode proteins with a helix-turn-helix DNA-binding motif. The conservation of aca genes has served as a signpost for the identification of acr genes, but the function of the proteins encoded by these genes has not been investigated. Here we reveal that an acr-associated promoter drives high levels of acr transcription immediately after phage DNA injection and that Aca proteins subsequently repress this transcription. Without Aca activity, this strong transcription is lethal to a phage. Our results demonstrate how sufficient levels of Acr proteins accumulate early in the infection process to inhibit existing CRISPR-Cas complexes in the host cell. They also imply that the conserved role of Aca proteins is to mitigate the deleterious effects of strong constitutive transcription from acr promoters.</AbstractText>
                <CopyrightInformation>Copyright © 2019 Elsevier Inc. All rights reserved.</CopyrightInformation>
            </Abstract>
            <AuthorList CompleteYN="Y">
                <Author ValidYN="Y">
                    <LastName>Stanley</LastName>
                    <ForeName>Sabrina Y</ForeName>
                    <Initials>SY</Initials>
                    <AffiliationInfo>
                        <Affiliation>Department of Molecular Genetics, University of Toronto, Toronto, ON M5S 1A8, Canada.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Borges</LastName>
                    <ForeName>Adair L</ForeName>
                    <Initials>AL</Initials>
                    <AffiliationInfo>
                        <Affiliation>Department of Microbiology and Immunology, University of California, San Francisco, San Francisco, CA 94143, USA.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Chen</LastName>
                    <ForeName>Kuei-Ho</ForeName>
                    <Initials>KH</Initials>
                    <AffiliationInfo>
                        <Affiliation>The J. David Gladstone Institutes, San Francisco, CA 94158 USA.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Swaney</LastName>
                    <ForeName>Danielle L</ForeName>
                    <Initials>DL</Initials>
                    <AffiliationInfo>
                        <Affiliation>The J. David Gladstone Institutes, San Francisco, CA 94158 USA; Quantitative Biosciences Institute, University of California, San Francisco, San Francisco, CA 94143, USA; Department of Cellular and Molecular Pharmacology, University of California, San Francisco, San Francisco, CA 94143, USA.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Krogan</LastName>
                    <ForeName>Nevan J</ForeName>
                    <Initials>NJ</Initials>
                    <AffiliationInfo>
                        <Affiliation>The J. David Gladstone Institutes, San Francisco, CA 94158 USA; Quantitative Biosciences Institute, University of California, San Francisco, San Francisco, CA 94143, USA; Department of Cellular and Molecular Pharmacology, University of California, San Francisco, San Francisco, CA 94143, USA.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Bondy-Denomy</LastName>
                    <ForeName>Joseph</ForeName>
                    <Initials>J</Initials>
                    <AffiliationInfo>
                        <Affiliation>Department of Microbiology and Immunology, University of California, San Francisco, San Francisco, CA 94143, USA; Quantitative Biosciences Institute, University of California, San Francisco, San Francisco, CA 94143, USA.</Affiliation>
                    </AffiliationInfo>
                </Author>
                <Author ValidYN="Y">
                    <LastName>Davidson</LastName>
                    <ForeName>Alan R</ForeName>
                    <Initials>AR</Initials>
                    <AffiliationInfo>
                        <Affiliation>Department of Molecular Genetics, University of Toronto, Toronto, ON M5S 1A8, Canada; Department of Biochemistry, University of Toronto, Toronto, ON M5S 1A8, Canada. Electronic address: alan.davidson@utoronto.ca.</Affiliation>
                    </AffiliationInfo>
                </Author>
            </AuthorList>
            <Language>eng</Language>
            <PublicationTypeList>
                <PublicationType UI="D016428">Journal Article</PublicationType>
            </PublicationTypeList>
            <ArticleDate DateType="Electronic">
                <Year>2019</Year>
                <Month>08</Month>
                <Day>27</Day>
            </ArticleDate>
        </Article>
        <MedlineJournalInfo>
            <Country>United States</Country>
            <MedlineTA>Cell</MedlineTA>
            <NlmUniqueID>0413066</NlmUniqueID>
            <ISSNLinking>0092-8674</ISSNLinking>
        </MedlineJournalInfo>
        <CitationSubset>IM</CitationSubset>
        <KeywordList Owner="NOTNLM">
            <Keyword MajorTopicYN="N">CRISPR-Cas</Keyword>
            <Keyword MajorTopicYN="N">Horizontal gene transfer</Keyword>
            <Keyword MajorTopicYN="N">Phage</Keyword>
            <Keyword MajorTopicYN="N">Pseudomonas aeruginosa</Keyword>
            <Keyword MajorTopicYN="N">Transcriptional regulator</Keyword>
            <Keyword MajorTopicYN="N">anti-CRISPR</Keyword>
        </KeywordList>
    </MedlineCitation>
    <PubmedData>
        <History>
            <PubMedPubDate PubStatus="received">
                <Year>2018</Year>
                <Month>12</Month>
                <Day>04</Day>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="revised">
                <Year>2019</Year>
                <Month>05</Month>
                <Day>06</Day>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="accepted">
                <Year>2019</Year>
                <Month>07</Month>
                <Day>25</Day>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="entrez">
                <Year>2019</Year>
                <Month>9</Month>
                <Day>3</Day>
                <Hour>6</Hour>
                <Minute>0</Minute>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="pubmed">
                <Year>2019</Year>
                <Month>9</Month>
                <Day>3</Day>
                <Hour>6</Hour>
                <Minute>0</Minute>
            </PubMedPubDate>
            <PubMedPubDate PubStatus="medline">
                <Year>2019</Year>
                <Month>9</Month>
                <Day>3</Day>
                <Hour>6</Hour>
                <Minute>0</Minute>
            </PubMedPubDate>
        </History>
        <PublicationStatus>aheadofprint</PublicationStatus>
        <ArticleIdList>
            <ArticleId IdType="pubmed">31474367</ArticleId>
            <ArticleId IdType="pii">S0092-8674(19)30846-3</ArticleId>
            <ArticleId IdType="doi">10.1016/j.cell.2019.07.046</ArticleId>
        </ArticleIdList>
    </PubmedData>
</PubmedArticle>
"""
