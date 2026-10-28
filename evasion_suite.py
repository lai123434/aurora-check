# -*- coding: utf-8 -*-
"""AI 生成文本压力测试套件（极光查重漏判诊断）。

这些文本全部由 AI 直接生成，覆盖从「典型机翻腔」到「高度拟人」的不同
拟人程度。目的：逼出引擎漏判（final_score < 0.58 却实为 AI）的薄弱环节。
"""

# 1) 典型机翻/官方腔 AI：结构化、绝对正确、密集连接词 —— 预期被抓住
S1_OBVIOUS_AI = """
It is worth noting that the rapid development of artificial intelligence has brought about multifaceted challenges across various domains. Moreover, while technology offers unprecedented convenience, it simultaneously raises profound ethical concerns that cannot be ignored. Firstly, the issue of privacy has become increasingly prominent. Secondly, the displacement of human labor demands careful consideration. In addition, the environmental cost of training large models should not be underestimated.

Delving deeper, we can observe that the implementation of AI systems requires a delicate balance between innovation and regulation. On the one hand, excessive restrictions may stifle progress. On the other hand, a complete lack of oversight could lead to catastrophic consequences. Therefore, a comprehensive framework is essential to ensure sustainable development.

In conclusion, artificial intelligence represents a double-edged sword that necessitates nuanced understanding. Only through collaborative efforts can we harness its potential while mitigating its risks. The future remains promising yet uncertain.
"""

# 2) 口语化、第一人称、带不完美 —— 拟人迷惑型，可能漏判
S2_CASUAL_AI = """
So I was talking to my friend the other day about this whole AI thing and honestly? I think people are way too scared of it. Like, yeah, it's gonna change stuff, but every new tech does that. My grandma used to say the microwave would kill us all. It didn't.

What bugs me is how everyone acts like it's this huge mystery. It's just math, right? Pattern matching. My little cousin uses it to do his homework and his grades actually went up, which is kinda wild when you think about it. Not that I'm saying kids should cheat, but the tool itself isn't evil.

Anyway I'm probably overthinking this. The point is we'll figure it out like we always do. People adapt. We always have.
"""

# 3) 学生风格注释条目：自然、像 90 分大一新生手写 —— 灰区，最危险
S3_STUDENT_AI = """
This article really helped me understand why some companies care more about soft skills than technical ones. The authors did a survey with 200 graduates and found that communication was the thing managers complained about most. I found this surprising because in class we always focus on numbers and models.

I will use this idea in my own essay about how students can prepare for jobs. The part about internships was useful since it shows real experience matters more than grades sometimes. My topic is about how students can develop skills, so this article gives me a good example to talk about.

One thing I disagreed with was the claim that everyone needs to learn coding. Not all jobs need it and the article didn't really prove this point well. But overall it was a clear and helpful source.
"""

# 4) 带"人味"瑕疵的 AI：跑题、重复、情绪化、小语法错 —— 高拟人
S4_IMPERFECT_AI = """
I keep thinking about this and I dont know maybe its just me but the whole remote work debate feels fake. Companies say they want flexibility but then they track your mouse movements?? That's not flexibility thats just surveillance with extra steps.

We tried it at my old job and it was actually fine?? People got more done. Meetings went down. But then leadership panicked because they couldnt see us sitting at desks and suddenly we're all back in the office staring at walls. Make it make sense.

Honestly the research on this is so mixed. One study says productivity up, another says down, another says it depends. Depends on what!!! Just say you dont know. The truth is nobody has figured it out and pretending we have is the real problem here.
"""

# 5) 自然议论文：像人写的评论，但其实是 AI —— 最难
S5_NATURAL_ESSAY = """
There's a comfortable story we tell ourselves about talent: that the people at the top earned it through sheer effort, and that anyone willing to work hard enough could get there too. I used to believe it. Most of us do, because it's reassuring. It means the world is fair.

But the more I look at how careers actually unfold, the less that story holds. The violinist who "practiced more" also had parents who could afford lessons. The founder who "took the risk" had a safety net to fall back on. Effort matters, of course it does, but it is not the whole equation, and pretending otherwise lets the lucky off the hook.

None of this is an argument against trying. It's an argument for being honest about the dice we're each handed. We can build a fairer system without first convincing ourselves the current one was earned.
"""

SAMPLES = {
    "S1_obvious_ai": S1_OBVIOUS_AI,
    "S2_casual_ai": S2_CASUAL_AI,
    "S3_student_ai": S3_STUDENT_AI,
    "S4_imperfect_ai": S4_IMPERFECT_AI,
    "S5_natural_essay": S5_NATURAL_ESSAY,
}

if __name__ == "__main__":
    import aurora_check as ac
    import os

    # 真实人类对照组：老师认可的 90 分真人学生终稿（非 AI 生成）
    HUMAN_CONTROLS = {}
    _final = "/workspace/BUS100_A1_Annotated_Bibliography_Final.docx"
    if os.path.exists(_final):
        try:
            HUMAN_CONTROLS["H1_real_student_final"] = ac.extract_text_from_file(_final)
        except Exception as e:
            print("提取人类对照失败:", e)

    print("=" * 70)
    print("AI 生成文本压力测试 — 引擎漏判诊断（含真实人类对照）")
    print("=" * 70)

    def run_one(name, txt, truth):
        res = ac.run_ai_detection(txt)
        fs = res.ai_probability
        label = res.label
        md = res.ml_detail or {}
        print(f"\n### {name}  [真实={truth}]")
        print(f"  final_score = {fs:.3f}  label = {label}  (ml_n={md.get('n')}, spread={md.get('spread'):.2f})")
        if md.get("scores"):
            print(f"  ML分项: " + ", ".join(f"{k.split('/')[-1]}={v:.2f}" for k, v in md["scores"].items()))
        if res.ml_disagreement:
            print("  ⚠ 多检测器强烈分歧 → 降级 mixed")
        es = res.engine_scores
        ranked = sorted(es.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)[:10]
        for k, v in ranked:
            print(f"    {k:24s} {v:.2f}")
        if truth == "AI" and fs < 0.58:
            print("  <<< 漏判(实为AI但分<0.58)")
        if truth == "human" and fs >= 0.58:
            print("  <<< 误杀(真人被判AI)")

    for name, txt in SAMPLES.items():
        run_one(name, txt, "AI")
    for name, txt in HUMAN_CONTROLS.items():
        run_one(name, txt, "human")

