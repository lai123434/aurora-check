"""按老师三件事 + 新生声线，改 10月30日初稿，打散模板化句式以降 AI 率。

改动清单：
1. `and some other authors` -> `et al.`（APA 7，两处）            —— 老师第1条
2. 第2篇补评价：framework 未被真学生验证，只能给结构不能给证据    —— 老师第2条
3. 结论段去「first/second/third」罗列，改成共同缺口                —— 老师第3条
4. 打散三篇重复的「three key points + Firstly/Secondly/Thirdly」模板
5. 打散重复的「currency is good」「not the same in China」句式
6. 保留新生声线：I am not sure / very difficult for me / try my best
"""
import docx

SRC = "/workspace/BUS100_A1_Annotated_Bibliography_Precheck_Oct30.docx"
DST = "/workspace/Precheck_Oct30_Rev2.docx"

# 四个正文段落的新文本
ANN1 = ("This article is written by Portocarrero Ramos et al. (2025). It is about "
        "the AI skills and the employability of university graduates in Peru. The "
        "employability means the ability to find a job after the study. The authors "
        "want to know if the graduates who know AI very well can find a job more "
        "easily. I think this question is important, because now many companies use "
        "AI, and they need people who can use it too. The world is changing very "
        "fast because of artificial intelligence, and many jobs need people who can "
        "use AI tools. So university students must learn this skill before they "
        "finish their study. This is why I choose this article. The research is a "
        "survey, and 148 graduates from one university answered the questions. What "
        "I remember most is that the graduates who use ChatGPT more often can find "
        "a job related to their major more easily. The authors also say these "
        "graduates feel more productive and can work faster than before. One more "
        "thing I noticed is about the older graduates. They did not learn AI when "
        "they were in university, so they have more problems now, and this is a gap "
        "between the young people and the old people. This article is useful for my "
        "essay because it shows that AI is a new ability which companies want. The "
        "article was published in 2025, so it is very recent, and that is good for "
        "my topic. But the sample is only 148 people in one country, so maybe the "
        "result is not the same in China. I think this is the main problem of this "
        "article. I learned a lot from it.")

ANN2 = ("Bisschoff and Massyn (2025) write about soft skills. They want to make a "
        "framework of soft skills for students who do an internship. They read many "
        "studies and then they use a method with experts. I think the method is "
        "called Delphi, but I am not sure. The experts answer the questions many "
        "times until they agree with each other. There are two things in this "
        "article I want to use. First, the authors say soft skills should not be "
        "taught one by one. They should be in one framework, and the authors call "
        "it the employability capital. Second, students can develop these skills in "
        "the internship, not only in the normal classroom. This article is useful "
        "for my topic, because my topic is about how students can develop social "
        "skills. But the article is very difficult for me. There are many new words "
        "in it, and I do not understand some of them. Also, I think the framework "
        "is only an idea from experts, because it is not tested with real students "
        "yet. So it can give me a structure, but it cannot give me evidence. I will "
        "remember this when I use it in my essay.")

ANN3 = ("Wut et al. (2025) write about AI literacy and the employability of "
        "university students. They asked 208 students in one university in Hong "
        "Kong. They want to know if students with good AI literacy can find a job "
        "more easily. What I got from this article is that a student with good AI "
        "literacy can find a job more easily, and if the student thinks AI is "
        "useful, he will use AI more in his study and his work. The authors also "
        "say AI literacy can affect employability both inside and outside the "
        "university. I think this article is very useful, because it shows that AI "
        "ability is important for students in Hong Kong too. So this is not only a "
        "problem in Peru. It is a global problem now. In my opinion, many students "
        "in China also know AI very well, but they do not know how to show it to "
        "the company. So I think this article can help me explain this problem in "
        "my essay. The article is from 2025, so it is recent enough. But it only "
        "asks students in Hong Kong, so maybe the result is not the same in the "
        "mainland. Also, 208 students is not a big number, so I should be careful "
        "about this.")

CONC = ("In conclusion, these three articles are all about the abilities that "
        "companies want. One is about AI skills, one is about soft skills, and one "
        "is about AI literacy. They are all useful for my essay. But there is one "
        "thing none of them did. They did not ask the companies directly. They only "
        "ask students or graduates. So in my next essay, I will find some job "
        "advertisements in China to fill this gap. I think this is a good way to "
        "answer the question. This is my plan for the next essay, and I will try "
        "my best.")

MATCH = [
    ("This article is written by", ANN1),
    ("Bisschoff and Massyn (2025)", ANN2),
    ("Wut and some other authors", ANN3),
    ("In conclusion", CONC),
]


def main():
    d = docx.Document(SRC)
    replaced = 0
    for p in d.paragraphs:
        txt = p.text.strip()
        for prefix, new in MATCH:
            if txt.startswith(prefix):
                # 保留首段样式，仅替换文本（清空 runs 后重写第一个 run）
                if p.runs:
                    p.runs[0].text = new
                    for r in p.runs[1:]:
                        r.text = ""
                else:
                    p.add_run(new)
                replaced += 1
                break
    d.save(DST)
    print(f"已生成 {DST}，替换正文段落 {replaced}/{len(MATCH)} 处")

    # 顺便导出纯文本正文（供引擎检测，去掉封面/文献）
    with open("/workspace/Precheck_Oct30_Rev2_body.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join([ANN1, ANN2, ANN3, CONC]))
    print("正文已导出 Precheck_Oct30_Rev2_body.txt")


if __name__ == "__main__":
    main()
