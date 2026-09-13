# 能理解语言、但无法产生自然言语的人群：研究动机与流行病学证据

> 用途：为 EEG-to-speech / speech neuroprosthesis 论文的 motivation、clinical significance 和 related work 提供可引用的证据。
>
> 检索与整理日期：2026-09-02。本文把“严格匹配的目标表型”和“可支持规模论证、但不能等同于目标表型的相关人群”明确分开。

## 1. 先给结论

目前没有一个可靠的中国、美国或全球统计项目，直接统计以下组合表型：

1. 能听懂别人讲话，至少保留实用的听觉/语言理解；
2. 认知和意图表达能力相对保留；
3. 不能产生可理解的自然言语，或完全不能发声；
4. 这种障碍主要来自运动输出通道、脑干/皮质延髓通路、喉/口面运动系统，而不是听力损失或语言理解障碍。

医学统计通常按病因或诊断编码，例如 locked-in syndrome（LIS）、anarthria、severe dysarthria、ALS、stroke 或 AAC need，而不是按“理解正常但说不出来”编码。因此，最稳妥的论文论证不是声称“全球有 X 百万人完全符合该表型”，而是：

> 这是一个临床上清晰、但流行病学上被分散记录和低估的人群。最严格的 LIS / anarthria 人群有直接病例和人群队列证据；更广义的严重运动性言语障碍和 AAC 需求人群规模更大，但不能把它们全部算作“理解正常且完全不能说话”。

这一区分不会削弱研究动机，反而使 argument 更可信：已有统计证明潜在人群不是单个病例，而现有统计又没有把最需要神经语音假体的人单独识别出来。

## 2. 表型定义：哪些人算，哪些人不能直接算

### 2.1 最严格的临床匹配：LIS 和 anarthria

经典 locked-in syndrome 的定义通常包括：意识清醒、四肢瘫痪、缄默/不能说话，而交流主要依赖垂直眼动和/或眨眼。挪威全国队列还要求严重交流障碍、四肢瘫痪或轻瘫、日常生活完全依赖他人，并且认知正常或接近正常。[Nilsen et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10491452/)

Anarthria 是“失去发音/构音能力”，更偏向运动言语输出障碍；它不等于 aphasia。最直接的神经假体证据来自一名脑干卒中后长期 anarthria 患者：研究报告其认知功能完整，并从皮质活动实时解码其尝试说出的词和句子。[Moses et al., 2021, *New England Journal of Medicine*](https://doi.org/10.1056/NEJMoa2027540) 这与本研究问题高度一致。

### 2.2 本项目应优先关注的失语症表型

严格来说，**失语症（aphasia）不是单纯的“不会说话”**，而是获得性语言障碍，可能影响语言表达、听理解、阅读和书写。[NIDCD](https://www.nidcd.nih.gov/health/aphasia) 因此，用户所描述的“能听懂话，但是说不了话”不是全部失语症，而是失语症中以**表达/口语输出受损、听理解相对保留**的一组表型。

最接近的临床术语是：

- **Broca’s aphasia / expressive aphasia / nonfluent aphasia：** 美国国家医学图书馆 MeSH 将其定义为表达性语言受损、接受性语言能力（包括理解）相对保留。[NLM MeSH: Aphasia, Broca](https://meshb.nlm.nih.gov/record/ui?dcmsLinks=true&ui=D001039)
- **Transcortical motor aphasia：** 自发言语少、非流利，理解相对保留，复述可能比 Broca 型更好；因此也可能符合“知道别人说什么，但自己很难说出完整话”的体验。
- **Severe nonfluent aphasia / speechless aphasia：** 口语输出可能只剩单词、固定语句、刻板语或近乎缄默；理解能力仍需逐项测量，不能仅凭“不会说”判断。

Broca 型并不意味着理解能力完全正常。NIDCD 指出，这类患者通常比 Wernicke 型更容易理解语言，但仍可能在复杂的口头、书面或手语语言结构上存在困难；其口语往往短促、费力、非流利。[NIDCD aphasia overview](https://www.nidcd.nih.gov/health/aphasia) 所以本项目在临床上应使用 **preserved or sufficiently functional auditory-language comprehension**，而不是未经测量的 “normal comprehension”。

### 2.3 相关但不能直接等同的人群

- **Severe dysarthria**：说话速度、清晰度或构音严重下降；一部分人仍能发声，因此不能全部算作“说不出来”。
- **ALS/MND**：病程后期可能出现严重 dysarthria 或失去自然言语，但 ALS 也可能伴随认知、行为和语言障碍，所以“能听懂话”必须单独测量，不能从 ALS 诊断自动推断。
- **卒中后失语（aphasia）**：可能同时损害理解和表达；不能把全部 aphasia 病例作为本研究目标。
- **喉切除或声带损伤**：可能保留语言理解而失去自然声音，但很多患者仍可使用假声、电子喉或其他替代发声；其神经机制和 EEG 解码目标与脑干卒中/LIS 不完全相同。
- **AAC 使用者**：是最接近“无法依赖自然说话”的广义服务人群，但 AAC 也服务于理解障碍、发育障碍和多种混合障碍，因此是上界/相关需求估计，不是严格表型人数。

## 3. 失语症：本项目应优先使用的核心临床人群

### 3.1 失语症与本研究问题的功能对应

失语症最重要的地方在于，它可以造成**语言表达通道与语言理解通道之间的不对称损害**。对于 Broca/非流利型失语症，患者可能清楚地知道自己想表达什么，也能相对理解他人讲话，但只能说出非常短、缓慢、费力或不完整的句子。这个“表达受限、理解相对保留”的组合，正是 EEG-to-speech 研究最应该关注的临床动机之一。

需要同时区分：

1. **语言表达障碍（aphasia）：** 词汇检索、句法组织、语音编码和口语/书写表达受损；
2. **运动性言语障碍（dysarthria）：** 已经形成的语言内容难以通过口面肌肉准确执行；
3. **言语失用（apraxia of speech）：** 言语动作计划和排序受损；
4. **听理解障碍：** 对词、句子或复杂语义的理解受损，常见于 Wernicke 型或 global aphasia。

同一位卒中患者可以同时有 aphasia、dysarthria 和 apraxia。因此，“能理解但说不了”应被作为一个**功能表型**来定义，而不能只用一个传统 aphasia subtype 名称代替。

### 3.2 美国、全球和中国的失语症规模

| 地区/人群 | 统计数字 | 对本项目的意义 | 主要限制 |
|---|---:|---|---|
| 美国，全部 aphasia | 约 **2 million** 人正在生活于 aphasia 中；NIDCD 还指出约三分之一卒中幸存者发生 aphasia | 说明美国存在百万量级的语言障碍人群，其中包含大量表达受损者 | 这是全部 aphasia，不是“理解保留且不能说话”的 Broca/非流利亚组；数字由 NIDCD 引用 National Aphasia Association，非专门的全国 subtype registry。[NIDCD](https://www.nidcd.nih.gov/health/aphasia) |
| 全球，卒中后 aphasia | 系统综述中，混合型卒中急性期中位频率约 **30%**，康复期约 **34%**；另一项覆盖 43 个国家的综述报告各研究为 **7%–77%**；2024 年 meta-analysis 汇总的总体 PSA 比例约 **34%** | 约三分之一卒中患者在某个阶段有 aphasia，说明潜在病因池很大 | 比例受卒中类型、评估时间、量表和诊断阈值影响；不能直接转成“不能说话”人数。[Ellis et al., 2016](https://doi.org/10.1016/j.apmr.2016.03.006)；[Frederick et al., 2022](https://doi.org/10.1044/2022_PERSP-22-00111)；[BMC Geriatrics meta-analysis, 2024](https://doi.org/10.1186/s12877-024-04765-0) |
| 全球，每年新发卒中相关 aphasia | 一项 NIHR 系统综述背景估计：全球每年约 10.3 million 新卒中中，约 **3.6 million（35%）** 伴随卒中相关 aphasia；约 **61%** 在一年后仍有交流问题 | 这比“单个罕见病例”的论证更能说明长期通信需求的规模 | 该换算基于较早的全球卒中年发病估计和广义 language impairment；不等于当前全球每年新增 severe nonfluent aphasia。[Brady et al., 2022](https://www.ncbi.nlm.nih.gov/books/NBK584825/?report=printable) |
| 中国，卒中后 aphasia | 中国卒中后失语专家共识称首次卒中后发生率可达 **32%**；中国临床调查综述引用急性期约 **15%–42%**、慢性期约 **25%–50%** | 说明中国同样有大规模 PSA 需求，且慢性阶段仍有相当患者遗留语言交流障碍 | 这些是共识/文献范围，不是中国全国 aphasia subtype 注册数据；没有直接给出“理解保留但不能说话”的人数。[中国卒中后失语临床管理专家共识摘要](https://drugs.dxy.cn/pc/clinicalGuidelines/yt3vcwQGzo9a_Cv4EMpYhZA)；[SPEECH study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8183701/) |
| 美国，急性缺血性卒中住院患者 | 2003–2014 年 National Inpatient Sample 中 4,339,156 例 AIS，**16.93%** 被记录有 aphasia；记录比例从 13.34% 上升到 21.94% | 提供了美国大规模真实世界的 aphasia 规模证据 | 行政数据库可能漏诊轻型/未筛查 aphasia；分母是住院 AIS，不是全体卒中幸存者；没有 Broca 或理解保留子集。[Wu et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32173230/) |

### 3.3 “能听懂但说不了”在 aphasia 中占多少？

这一问题目前没有可靠的全国或全球直接答案。原因是多数数据库只记录“是否 aphasia”，而不同时保留听理解、口语流利度、命名、复述、阅读和书写等完整维度；而且失语症亚型会在卒中后康复过程中变化。[Brady et al., 2022](https://www.ncbi.nlm.nih.gov/books/NBK584825/?report=printable)

现有亚型研究只能提供**方向性参照**：

- Copenhagen aphasia study 报告，首次卒中急性期 aphasia 患者中，Broca 型约占 **12%**，global 型约占 32%；一年后 Broca 型约占 13%。[Pedersen et al., 2004](https://pubmed.ncbi.nlm.nih.gov/14530636/)
- 另一项亚急性卒中研究中，625 名 aphasia 患者里 170 名（**27.2%**）被归为 Broca 型；不同研究的亚型比例差异较大。[The spectrum of aphasia subtypes and etiology in subacute stroke](https://pubmed.ncbi.nlm.nih.gov/23680689/)
- 一项早期连续急性卒中队列中，850 名卒中患者有 177 名 aphasia；其中 9 人被归为 Broca 型且 gross comprehension 保留，另有 4 人为正常理解的 nonfluent anomic aphasia。13/850 约 **1.5%** 是一个“非流利且理解相对保留”的历史性近似比例，但这些患者并不一定完全不能说话，且研究使用的是 1976 年的临床分类，不能当作今天的患病率。[Brust et al., 1976](https://pubmed.ncbi.nlm.nih.gov/1265809/)

因此，不能严谨地把“全球 aphasia 患者 × Broca 比例”写成“全球能听懂但说不了话的人数”。尤其是**严重到完全没有可理解自然言语**的患者，只是 Broca/非流利型中的更小子集，现有流行病学研究没有统一分母。

### 3.4 对本项目最重要的结论

对 EEG-to-speech 研究，最合理的目标表型可以写成：

> **People with preserved or sufficiently functional auditory-language comprehension and severely impaired, effortful, or absent intelligible natural speech, including severe nonfluent/expressive aphasia and aphasia complicated by apraxia or dysarthria.**

中文可写为：

> **保留相对可用的听觉—语言理解能力，但自然言语严重受损、费力、不可理解或缺失的人群，包括重度非流利/表达性失语，以及合并言语失用或构音障碍的失语症患者。**

这一定义比“所有 aphasia”更贴近患者真正的通信困境，也比只使用“Broca aphasia”更适合研究，因为临床表现和病灶并不总是严格对应传统亚型。

## 4. 直接匹配表型的数字证据

| 人群/地区 | 研究或统计数字 | 与目标表型的关系 | 不能过度解读之处 |
|---|---:|---|---|
| 挪威，血管性 long-lasting LIS | 2012–2022 年全国登记队列 51 人；43 人有随访；23 人后来脱离 LIS；最后登记状态仍在 LIS 的 16 人 | 该队列要求严重交流障碍、四肢瘫痪/轻瘫和正常或接近正常认知，是最接近“理解/意识保留但不能说”的人群级证据之一 | 51 是十年累积队列，不是点患病率；16 是随访时仍处于 LIS 状态者，且全部属于 incomplete LIS；不能直接外推全球人数。队列中 43 人随访时只有 1 人仍为 classic LIS，7 人无 verbal response，12 人只能发声，16 人有 dysarthric speech，8 人有功能性言语。[Nilsen et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10491452/) |
| 挪威，数量级换算 | 16 / 5,425,270 ≈ 2.95 人/百万人 | 作为长期 LIS 状态的数量级参照 | 这是根据队列状态和论文给出的 2021 年人口数计算的“方向性比值”，不是作者报告的正式点患病率；严格 classic LIS 还要更少，且病例筛选和随访定义会影响分母/分子 |
| 荷兰，护理机构 | 187 个长期照护机构调查，91.4% 回复；确认 10 例 LIS，其中 classic LIS 的点患病率为 0.7/10,000 张 somatic nursing-home beds | 提供了 classic LIS 的正式点患病率证据 | 分母是护理院床位，不是普通人口；不能直接写成“荷兰普通人口 7/100,000”。[Kohnen et al., 2013](https://pubmed.ncbi.nlm.nih.gov/23306659/) |
| 美国，脑干卒中后 anarthria | 1 名长期 anarthria、痉挛性四肢轻瘫患者；认知功能完整；可实时从皮质信号解码句子，15.2 词/分钟，中位词错误率 25.6% | 是“理解/认知仍在但自然言语输出消失”的直接临床 proof-of-concept | 单病例脑机接口研究，不是流行病学估计；但是对本项目的临床可行性和需求论证非常直接。[Moses et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8972947/) |

**解释重点：** 挪威和荷兰的数据说明，严格 LIS 是真实存在、但非常罕见且定义不统一的表型。尤其是“classic LIS”会随康复、运动恢复和随访时间变化，所以“某一时点仍不能说话”的人数不能简单用初诊病例数代替。

## 5. 美国、中国和全球的相关规模证据

### 5.1 ALS：最接近长期、进行性失去言语输出的病因之一

ALS 的全国/全球统计可以帮助建立潜在患者池，但不能直接等同于目标表型。

- **美国：** CDC/ATSDR 的 National ALS Registry 估计 2022 年美国约有 33,000 名 ALS 患者，并预测 2030 年超过 36,000 名。[CDC/ATSDR, 2025](https://www.cdc.gov/als/php/abstracts-publications-reports/prevalence-2022-2030.html)
- **ALS 中的言语障碍：** 一项 88 人 ALS 临床队列用 Sentence Intelligibility Test 量化发现，78% 有 dysarthria；既往报告范围为 33%–93%。但该比例表示可量化的说话清晰度/速度障碍，不表示 78% 都完全不能说话。[Profiles of Dysarthria and Dysphagia in ALS](https://pmc.ncbi.nlm.nih.gov/articles/PMC10023186/)
- **言语丧失的时间进程：** 对 166 名 ALS 患者的纵向研究估计，bulbar-onset ALS 达到 speech loss 阈值的时间约为：按说话速度 <120 words/min 定义为 23 个月，按可理解度 <85% 定义为 32 个月；spinal-onset 患者的功能性言语通常维持更久。[Eshghi et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9489769/)
- **美国/全球认知限制：** ALS 不是“认知理解必然正常”的同义词。ALS 可伴随语言、执行功能和行为变化，因此研究招募时应实测听觉理解和认知，而不是按 ALS 病例数直接相乘。[ALS cognition review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746914/)

**中国：** 一项全国性城市医保数据库研究覆盖约 4.3 亿人，报告 2016 年 ALS 粗患病率 2.91/100,000 person-years，按 2010 年中国人口普查标准化后的患病率为 2.97/100,000；研究对象主要是城镇职工和城镇居民医保人群。[Xu et al., 2020, *JNNP*](https://pubmed.ncbi.nlm.nih.gov/32139654/) 该研究没有报告“不能说话且理解保留”的子集，因此不能把 2.97/100,000 直接转换成目标患者数。

**全球：** 一项系统综述和 meta-analysis 汇总得到全球 ALS 粗患病率约 4.42/100,000，发病率约 1.59/100,000 person-years。[Xu et al., 2020](https://pubmed.ncbi.nlm.nih.gov/31797084/) 另一项 2023 年系统综述强调，各国研究覆盖和登记质量差异很大，全球估计存在显著异质性。[Wolfson et al., 2023](https://doi.org/10.1212/WNL.0000000000207474) 这些数字可以作为“ALS 潜在池”的背景，但不是“保持理解、失去言语”的流行病学数字。

### 5.2 卒中：人数很大，但必须从总卒中中识别运动性言语障碍

- **全球：** WHO/GBD 2021 估计 2021 年全球有约 93.8 million prevalent stroke cases 和 11.9 million new stroke cases。[WHO, 2025](https://www.who.int/news-room/fact-sheets/detail/stroke)
- **中国：** 中国成人卒中监测/估计报告 2020 年约有 17.8 million 名 40 岁以上成年人经历过卒中，约 3.4 million 为当年首次卒中。[Wang et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36862407/) 这是卒中总人群，不是不能说话的人数。
- **美国：** CDC 的 PLACES 页面依据 NHANES 2017–2020 估计约 9.4 million 美国成年人曾被医生告知患过卒中。[CDC PLACES](https://www.cdc.gov/places/measure-definitions/health-outcomes.html) 旧版 NHIS 2018 的官方数字为 7.8 million，因调查年份和方法不同，不应与 9.4 million 混用。[CDC NCHS FastStats](https://www.cdc.gov/nchs/fastats/stroke.htm)
- **卒中后的 dysarthria：** 文献对卒中后 dysarthria 的估计大约为 20%–42% 的卒中幸存者，部分临床资料在急性期报告 22%–58%；但 dysarthria 可能是轻度或中度，不能把这个比例当成 anarthria。[Neuroanatomical regions associated with non-progressive dysarthria post-stroke, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9479301/)
- **区分 aphasia 与 dysarthria：** 在英国 88,974 名卒中幸存者的二次分析中，24% 被记录为 dysarthria only，28% 同时有 aphasia 和 dysarthria，12% 为 aphasia only。[Taylor et al., 2020](https://doi.org/10.1080/02687038.2020.1759772) 这项研究不能直接给出“完全不能说话且理解正常”的人数，但说明“表达通道障碍”和“语言理解/语言系统障碍”在大规模数据中可以、也应该被区分。

**对论文的含义：** 卒中提供了数量巨大的潜在病因池，但只有脑干/皮质延髓运动通路损伤、严重构音障碍或 anarthria 的子集与本项目直接对应；需要把 comprehension、aphasia、dysarthria、anarthria 分开测量。

### 5.3 中国“言语残疾”数据：有全国数字，但不能直接当作目标表型

中国第二次全国残疾人抽样调查以 2006-04-01 为标准时点，推算大陆地区各类残疾总人数约 82.96 million，其中“言语残疾”约 1.27 million，占残疾人总数 1.53%。原始数据可见中国政府网发布的调查公报；具体分类数字也见地方残联转载的公报文本。[中国政府网公报](https://www.gov.cn/fuwu/cjr/2009-05/08/content_2630949.htm)；[公报数字转载](https://swsadmin.shanwei.gov.cn/sdpf/zsyd/200909/6fa51851969841649185f99c7a897eb8.shtml)

这个 1.27 million 是目前中国最接近“言语输出困难”全国规模的旧基线，但不能写成“1.27 million 中国人能听懂话却说不出来”，原因包括：

- “言语残疾”是行政/调查分类，不等于神经学上的 anarthria；
- 可能包含语言理解障碍、听力相关语言问题、发育性障碍、构音障碍和多重残疾；
- 数据是 2006 年调查推算，距今已近 20 年；
- 中国政府 2026 年宣布启动第三次全国残疾人抽样调查，预计覆盖约 2.8 million 人，调查持续到 2028 年；这说明全国层面的新分类和需求数据正在更新中。[中国政府网英文公报, 2026](https://english.www.gov.cn/policies/latestreleases/202605/29/content_WS6a18da27c6d00ca5f9a0b4fe.html)

因此，1.27 million 适合在论文中作为“广义言语功能障碍/潜在服务需求的历史上界或背景规模”，不适合当作严格目标表型估计。

### 5.4 AAC：最接近“无法依靠自然言语交流”的广义上界

美国 Speech-Language-Hearing Association（ASHA）指出，由于诊断、年龄、地域、交流方式和 AAC 使用程度高度异质，AAC 使用者的患病率很难准确估计。Beukelman and Light (2020) 的估计是：约 **5 million Americans**、约 **97 million people worldwide** 可能从 AAC 中获益。[ASHA AAC Practice Portal](https://www.asha.org/Practice-Portal/Professional-Issues/Augmentative-and-Alternative-Communication/)

这个数字对 motivation 很有用，但必须标注为 **broad upper-bound / service-need estimate**：AAC 同时覆盖自然言语缺失、严重言语障碍、发育性障碍、理解障碍及多重残疾；它不是“理解正常但完全不能说话”的统计人数。更合适的写法是“已有约 5 million 美国人、97 million 全球人口可能需要 AAC 支持，说明自然言语通道受限的社会需求规模远大于目前发表的脑机接口病例数”。

## 6. 最可辩护的数量级表述

### 严格表型

> **Exact prevalence is unknown.** 目前没有国家或全球数据库把“保留听觉/语言理解 + 丧失可理解自然言语”作为独立类别统计。经典 LIS 的人群证据显示它是罕见表型：荷兰护理机构的 classic LIS 点患病率为 0.7/10,000 张护理床位；挪威全国 long-lasting vascular LIS 队列在十年内登记 51 人，随访时 16 人仍处于 LIS 状态，但这些数字不能直接外推为普通人口或全球患病率。

### 临床相关的广义表型

> **The clinically relevant population is substantially larger than classic LIS, but no single validated count exists.** ALS、卒中后 severe dysarthria/anarthria 以及其他运动言语障碍共同构成更大的潜在人群；美国约 33,000 名 ALS 患者、全球约 93.8 million 名现患卒中者和 AAC 约 5 million 美国/97 million 全球受益者估计，都应被当作病因池或广义需求上界，而不是严格目标人群人数。

### 建议避免的写法

不要写：

> “There are 97 million people worldwide who understand speech but cannot speak.”

也不要把 1.27 million 中国“言语残疾”、33,000 美国 ALS 或 93.8 million 全球卒中直接改写成目标表型人数。

建议写：

> “A clinically important but epidemiologically undercounted population consists of people whose cognition and/or language comprehension remains available while natural speech is severely impaired or absent, including individuals with anarthria, severe dysarthria, locked-in syndrome, and advanced motor-neuron disease. No registry currently isolates this phenotype. Existing population-based studies nevertheless document a real and persistent need: classic LIS is rare but measurable, speech impairment is common in ALS and stroke populations, and millions of people worldwide may benefit from AAC. This gap motivates non-muscular speech-decoding approaches that can bypass the damaged vocal-motor output pathway.”

## 7. 为什么这足以支持 EEG-to-speech 研究

证据链可以写成四步：

1. **功能解离真实存在：** LIS 和脑干卒中后 anarthria 证明，意识、认知、语言意图可以保留，而自然语音运动输出可能消失。[Nilsen et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10491452/)；[Moses et al., 2021](https://doi.org/10.1056/NEJMoa2027540)
2. **病因池并不只有单病例：** ALS 和卒中是有大规模人群统计的病因池，且言语障碍在 ALS 和卒中幸存者中都很常见。[ALS speech cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC10023186/)；[stroke dysarthria review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9479301/)
3. **现有统计不适合直接回答本问题：** 诊断分类通常混合 comprehension、language、voice 和 motor-speech impairment，导致“能理解但不能说”的子集没有独立分母。
4. **技术目标明确：** 研究的价值不是替代所有 AAC，而是为那些仍有语言/意图、却无法可靠使用自然语音或常规肌肉控制接口的人，提供一个绕过 vocal-motor bottleneck 的通信通道。NEJM 单病例和后续 Nature speech neuroprosthesis 工作已经证明这种临床方向具有可行性。[Moses et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8972947/)；[Metzger et al., 2023](https://doi.org/10.1038/s41586-023-06443-4)

## 8. 对本项目样本/临床定义的建议

为了让后续数据和论文表述真正对应上述 argument，建议把受试者特征拆成三个独立维度，而不是只写“不能说话”：

- **理解：** auditory yes/no comprehension、词/句理解、指令执行；必要时使用不依赖口头回答的测验；
- **语言/意图：** 保留的词汇、句法和语义能力，以及是否能进行眼动、手势、书写或 AAC 交流；
- **输出：** natural speech intelligibility、speaking rate、是否 anarthric、是否只能产生非言语声音，以及是否存在可用的残余口面/喉部运动。

论文中可将目标人群写为：

> “people with preserved or sufficiently functional auditory-language comprehension and severely impaired or absent intelligible natural speech.”

这一定义比“speech-impaired patients”严格，也比只限定 classic LIS 更能覆盖 EEG-to-speech 的潜在临床应用，同时避免把 aphasia、听力损失和一般 AAC 需求混为一谈。

## 9. 证据—主张对应表

| 主张 | 主要证据 | 证据等级与限制 |
|---|---|---|
| 经典 LIS 具有清醒/认知保留、缄默和严重运动障碍的组合 | Nilsen et al. 2023 | 全国登记队列/原始研究；LIS 定义和随访状态存在异质性 |
| “理解/认知仍在但自然言语不能产生”具有直接临床实例 | Moses et al. 2021 NEJM | 直接病例证据；单病例，不能做患病率 |
| classic LIS 很罕见，且缺乏统一全球患病率 | Kohnen et al. 2013；Nilsen et al. 2023 | 荷兰分母是护理床位；挪威是病例队列，不是全球抽样 |
| ALS 言语障碍常见、且可随病程进展到 speech loss | ALS dysarthria cohort；Eshghi et al. 2022 | 临床队列/纵向研究；不等于所有 ALS 都完全不能说话，也不保证理解保留 |
| 中国 ALS 有全国性大样本疾病负担估计 | Xu et al. 2020 JNNP | 覆盖城市医保人群，不是所有中国居民；没有目标表型子集 |
| 卒中提供巨大潜在病因池，但需要区分 aphasia 与 dysarthria | WHO/GBD、CDC、中国卒中研究、卒中 dysarthria review | 总卒中人数很可靠；speech-specific fractions 跨研究差异较大 |
| 中国“言语残疾”至少有历史全国基线 | 2006 第二次全国残疾人抽样调查 | 官方历史行政分类；1.27m 不等于理解保留的 anarthria |
| 广义自然言语受限/AAC 服务需求远大于 LIS | ASHA 引用 Beukelman & Light 2020 | 5m/97m 是广义受益估计，不是严格患病率 |
| Broca/非流利型失语症可表现为表达受损、理解相对保留 | NLM MeSH；NIDCD | 定义性/临床权威证据；“相对保留”不等于所有复杂句理解正常 |
| 卒中后 aphasia 的总体比例约三分之一，但跨研究差异很大 | [2016 meta-analysis](https://doi.org/10.1016/j.apmr.2016.03.006)；[2022 全球综述](https://doi.org/10.1044/2022_PERSP-22-00111)；[2024 meta-analysis](https://doi.org/10.1186/s12877-024-04765-0) | 综述/Meta 分析；不能直接得到“完全不能说话”的比例 |
| 美国约 2m 人生活于 aphasia | NIDCD | 官方健康信息页引用 NAA；全部 aphasia，非 subtype registry |
| 中国 PSA 约 15%–42%（急性期）和 25%–50%（慢性期）的文献范围 | SPEECH study；中国专家共识 | 中国临床文献/共识；不是全国患者数据库，且没有理解保留子集 |
| “理解保留且非流利”没有统一流行病学分母 | NIHR aphasia report；Copenhagen/历史亚型研究 | 亚型会变化，评估工具和诊断标准不一；只能用于方向性参照 |

## 10. 检索说明与引用注意事项

本次优先使用 PubMed/PMC、NEJM、Neurology、JNNP、CDC、WHO/WSO、中国政府网和 ASHA 等可核查来源。中文 CNKI/万方的检索不应在本文件中被表述为“已完成系统综述”；后续若需要补充中文文献，可使用以下检索词：

- `闭锁综合征 患病率 / 流行病学`
- `脑卒中 构音障碍 患病率 / 言语清晰度`
- `失语 构音障碍 区分 卒中`
- `无言语患者 AAC / 辅助替代沟通`
- `肌萎缩侧索硬化 构音障碍 言语丧失`
- `能听懂 不能说 脑干卒中 / anarthria`

引用时应始终保留三个限定词：

1. **exact phenotype unknown**；
2. **underlying disease counts are not target counts**；
3. **AAC and broad speech-disability numbers are upper-bound/service-need estimates**。

## 11. 主要参考文献

1. Nilsen, H. W., et al. (2023). *Demographic, Medical, and Clinical Characteristics of a Population-Based Sample of Patients With Long-lasting Locked-In Syndrome*. **Neurology**, 101(10), e1025–e1035. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10491452/)
2. Kohnen, R. F., Lavrijsen, J. C. M., Bor, J. H. J., & Koopmans, R. T. C. M. (2013). *The prevalence and characteristics of patients with classic locked-in syndrome in Dutch nursing homes*. **Journal of Neurology**, 260, 1527–1534. [PubMed](https://pubmed.ncbi.nlm.nih.gov/23306659/) · [DOI](https://doi.org/10.1007/s00415-012-6821-y)
3. Moses, D. A., et al. (2021). *Neuroprosthesis for Decoding Speech in a Paralyzed Person with Anarthria*. **New England Journal of Medicine**, 385, 217–227. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8972947/) · [DOI](https://doi.org/10.1056/NEJMoa2027540)
4. Mehta, P., et al. (2025). *Amyotrophic lateral sclerosis estimated prevalence cases from 2022 to 2030, data from the National ALS Registry*. **Amyotrophic Lateral Sclerosis and Frontotemporal Degeneration**. [CDC/ATSDR summary](https://www.cdc.gov/als/php/abstracts-publications-reports/prevalence-2022-2030.html)
5. *Profiles of Dysarthria and Dysphagia in Individuals With Amyotrophic Lateral Sclerosis* (2023). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10023186/)
6. Eshghi, M., Yunusova, Y., Connaghan, K. P., et al. (2022). *Rate of speech decline in individuals with amyotrophic lateral sclerosis*. **Scientific Reports**, 12, 15713. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9489769/) · [DOI](https://doi.org/10.1038/s41598-022-19651-1)
7. Xu, L., et al. (2020). *Incidence and prevalence of amyotrophic lateral sclerosis in urban China: a national population-based study*. **JNNP**, 91(5), 520–525. [PubMed](https://pubmed.ncbi.nlm.nih.gov/32139654/) · [DOI](https://doi.org/10.1136/jnnp-2019-322317)
8. Xu, L., et al. (2020). *Global variation in prevalence and incidence of amyotrophic lateral sclerosis: a systematic review and meta-analysis*. **Journal of Neurology**, 267, 944–953. [PubMed](https://pubmed.ncbi.nlm.nih.gov/31797084/) · [DOI](https://doi.org/10.1007/s00415-019-09652-y)
9. Wolfson, C., et al. (2023). *Global Prevalence and Incidence of Amyotrophic Lateral Sclerosis: A Systematic Review*. **Neurology**, 101(6), e613–e623. [DOI](https://doi.org/10.1212/WNL.0000000000207474)
10. Taylor, E. K., et al. (2020). *Prevalence of aphasia and dysarthria among inpatient stroke survivors: describing the population and potential treatment needs*. [DOI](https://doi.org/10.1080/02687038.2020.1759772)
11. *Neuroanatomical regions associated with non-progressive dysarthria post-stroke: a systematic review* (2022). **BMC Neurology**, 22, 353. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9479301/)
12. World Health Organization. (2025). *Stroke fact sheet*. [WHO](https://www.who.int/news-room/fact-sheets/detail/stroke)
13. Centers for Disease Control and Prevention. *Stroke among adults / PLACES health outcomes*. [CDC](https://www.cdc.gov/places/measure-definitions/health-outcomes.html)
14. American Speech-Language-Hearing Association. *Augmentative and Alternative Communication (AAC)*. [ASHA Practice Portal](https://www.asha.org/Practice-Portal/Professional-Issues/Augmentative-and-Alternative-Communication/)
15. 中国政府网. *2006年第二次全国残疾人抽样调查主要数据公报*. [Gov.cn](https://www.gov.cn/fuwu/cjr/2009-05/08/content_2630949.htm)
16. National Institute on Deafness and Other Communication Disorders. *Aphasia*. [NIDCD](https://www.nidcd.nih.gov/health/aphasia)
17. National Library of Medicine. *Aphasia, Broca: MeSH Descriptor Data*. [MeSH](https://meshb.nlm.nih.gov/record/ui?dcmsLinks=true&ui=D001039)
18. Flowers, H. L., et al. (2016). *Poststroke Aphasia Frequency, Recovery, and Outcomes: A Systematic Review and Meta-Analysis*. **Archives of Physical Medicine and Rehabilitation**, 97, 2188–2201.e8. [DOI](https://doi.org/10.1016/j.apmr.2016.03.006)
19. Frederick, A., Jacobs, M., Adams-Mitchell, C. J., & Ellis, C. (2022). *The Global Rate of Post-Stroke Aphasia*. **Perspectives of the ASHA Special Interest Groups**, 7, 1567–1572. [DOI](https://doi.org/10.1044/2022_PERSP-22-00111)
20. Brady, M. C., et al. (2022). *Complex speech-language therapy interventions for stroke-related aphasia: the RELEASE study*. **Health and Social Care Delivery Research**, 10(28). [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK584825/?report=printable)
21. Wu, C., et al. (2020). *Prevalence and Impact of Aphasia among Patients Admitted with Acute Ischemic Stroke*. **Journal of Stroke and Cerebrovascular Diseases**, 29(5), 104764. [PubMed](https://pubmed.ncbi.nlm.nih.gov/32173230/) · [DOI](https://doi.org/10.1016/j.jstrokecerebrovasdis.2020.104764)
22. Pedersen, P. M., et al. (2004). *Aphasia after stroke: type, severity and prognosis. The Copenhagen aphasia study*. **Cerebrovascular Diseases**, 17(1), 35–43. [PubMed](https://pubmed.ncbi.nlm.nih.gov/14530636/)
23. Brust, J. C. M., et al. (1976). *Aphasia in acute stroke*. **Stroke**, 7(2), 167–174. [PubMed](https://pubmed.ncbi.nlm.nih.gov/1265809/) · [DOI](https://doi.org/10.1161/01.str.7.2.167)
24. Hoffmann, M., & Chen, R. (2013). *The spectrum of aphasia subtypes and etiology in subacute stroke*. **Journal of Stroke and Cerebrovascular Diseases**, 22(8), 1385–1392. [PubMed](https://pubmed.ncbi.nlm.nih.gov/23680689/) · [DOI](https://doi.org/10.1016/j.jstrokecerebrovasdis.2013.04.017)
25. *A physician survey of poststroke aphasia diagnosis and treatment in China: SPEECH study*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8183701/)
26. *Meta-analysis and systematic review of the relationship between sex and the risk or incidence of poststroke aphasia and its types* (2024). **BMC Geriatrics**. [DOI](https://doi.org/10.1186/s12877-024-04765-0) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/38438862/)
