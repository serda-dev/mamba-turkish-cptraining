# Türkçe Jamba2 CPT: araştırma ve ölçüme bağlı koşu planı

Araştırma tarihi: 6 Ekim 2026. Bu belge eğitim sonucu veya GPU hız ölçümü değildir. Handoff, mevcut konfigürasyon ve birincil kaynaklara dayanır; oranlar, token bütçeleri ve eşikler açıkça başlangıç mühendislik seçimleridir. Aşağıdaki seçenekler aynı anda zorunlu uygulamalar değildir.

## Başlangıç ağırlığını doğru tanımlama

`ai21labs/AI21-Jamba2-3B` ham pretrained base değildir. AI21'in yayımladığı Jamba2 eğitim açıklaması, mid-training sonrasında SFT, DPO ve RL kullanıldığını belirtir [S1]. Dolayısıyla bu ağırlığa düz metin CPT uygulamak, hazır talimat yeteneğini de değiştirecektir. Türkçe dil modelleme kaybının düşmesi, bu yeteneğin korunduğunu kanıtlamaz.

Güncel `serda-dev/Jamba2-3B-Turkish` model kartı kendisini CPT + QLoRA SFT sürümü olarak tanımlar. Aynı kart CPT checkpoint'i için yine aynı repo adını kullanır ve örneklerde `-SFT` adına geçer [S2]. Bu nedenle repo adına bakarak mevcut `main` ağırlığının temiz CPT olduğunu varsayamayız. Hangi revision'ın hangi aşamaya ait olduğu ağırlık/config/tokenizer/adapter manifestiyle doğrulanmalıdır. Yeni koşu, doğrulanmış temiz CPT checkpoint'i veya açıkça tanımlanan resmi Jamba2 checkpoint'inden başlamalıdır. Sonradan birleşmiş adapterı farkında olmadan başlangıç yapmak uygun değildir. Resmi checkpoint seçimi, modelin sıfırdan eğitileceği anlamına gelmez.

Resmi model config'inde `num_experts=1`, 28 katman ve `attn_layer_period=14`, `attn_layer_offset=7` bulunur [S3]. Bu 3B sürüm için genel Jamba MoE toplam/aktif parametre sayılarını kullanmayın. Config attention konumlarını 7 ve 21 olarak belirler; kalan katmanlar Mamba'dır. Mimari, düşük bellekli uzun bağlam çıkarımı için motivasyondur; tek GPU CPT hızının rakip modellerden belirli kat hızlı olacağını kanıtlamaz. Uzun bağlam kapasitesini kısa CPT'de otomatik koruduğumuzu da iddia etmeyin.

## Geçmiş sorunları ayırarak değerlendirme

Kullanıcının bildirdiği muhakeme/metin anlama ile format/talimat başarısı ayrımı, iki farklı ölçüm gerektirir. Yanıtın anlamsal doğruluğu ve geçerli JSON/talimat uyumu ayrı raporlanmalıdır. Mevcut konuşma bağlamındaki bir recovery koşusunda Gold v1 toplam başarısı 6/84 olarak bildirilmiştir; bu, önceki tüm modellerin muhakemesinin iyi olduğu veya yeni corpusun bu sorunu tek başına çözeceği anlamına gelmez. Bu eski diagnostic bir geçmiş gözlemdir; bu belge ham dosyanın güncel sürümünü yeniden okuyup doğrulamış değildir. Yeni baseline aynı testleri yeniden çalıştırmalıdır.

Araştırma, dil uyarlamasında İngilizce replay olmadan downstream/ICL yeteneklerinin Türkçe benzeri hedef dil loss'undan önce bozulabileceğini gösterir [S4]. Başka bir kapsamlı uyarlama çalışması geçerli çıktı sorunlarını ayrı ölçer [S5]. Bunlar Jamba2 üzerinde geçmiş arızanın kesin nedenini ispatlamaz. Olası nedenleri şu sırayla elemek daha ucuzdur:

1. Token ID, özel token, chat template, EOS, embedding boyutu veya adapter merge uyuşmazlığı.
2. Evaluation'ın yanlış prompt/parser/cevap kesme ayarı; format hatasıyla içerik hatasının karışması.
3. Çok yüksek LR, yanlış gerçek veri karışımı, biten shard'ın sessizce tekrar edilmesi, bozuk resume.
4. Corpus dağılımı nedeniyle talimat unutma ve yetersiz/uyumsuz SFT.

## İngilizce replay ve CPT veri dağılımı

İngilizce replay korunsun. ACL 2025 dil adaptasyon çalışması [S4] ve 2026 CoLLAs replay çalışması [S6] bunu destekler; ikincisi Llama ailesinde çalışır. Bu sonuçları Jamba/Türkçe için belirli bir optimum yüzde olarak aktarmak doğru değildir. Orijinal Jamba2 pretraining verisi erişilebilir değildir; açık İngilizce kaynaklarımız orijinal dağılımın tam replay'i değil, retention için bir vekildir.

İlk aday toplam eğitilen tokenlarda %80 Türkçe + %20 retention olsun. Bu önceki İngilizce ağırlığı yüksek koşuyu kopyalamayan ve hedef dili öne çıkaran bir hipotezdir. Retention içinde örnek dağılım %70 genel/eğitsel İngilizce, %20 matematik, %10 kod olabilir; bu toplamda %14 + %4 + %2 eder. Bu oranlar belge/GB oranı değil, tokenizer sonrasında gerçek eğitim tokenı oranıdır. Matematik ve kod doğal Türkçe kaynaklarda da bulunabilir; kaynak etiketi ile metnin dili aynı şey değildir.

İlk 100M token sonrasında İngilizce/format retention bozuluyorsa LR ve replay ayarı incelensin; örneğin %70/%30 yeni deney seçeneğidir. Türkçe gelişimi zayıf ama retention kararlıysa %85/%15 düşünülebilir. Bunları aynı koşuda gizlice değiştirerek bilimsel karşılaştırma saymayın; değişikliği checkpoint/token sayısıyla yeni run manifestine işleyin. İlk koşuda üç tam deneylik grid araması zorunlu değildir.

Türkçede başlangıç kabul grubu `PREMIUM` + `KEEP` olsun. Kaynak büyüklüğü kaliteyi kanıtlamaz; dil, content_type ve kaynak çeşitliliği izlenmeli, devasa web ailesi tek başına tüm bütçeyi tüketmemelidir. Sabit aile eşitliği de ölçümsüz zorunlu değildir. Cosmos için kabul örnekleri ayrıca denetlensin. `REVIEW`, `DETERMINISTIC_REPAIR`, `MODEL_REPAIR` ham haliyle temiz kabul edilmesin; onarım sonrası yeniden hash/kalite kontrolü ve köken kaydı olmadan eklenmesin. `DROP` eğitime girmesin.

Veri havuzunu dört Türkçe parçaya bölüp ilkinde genel İngilizce, sonra sadece matematik, sonrasında sadece kod vermek, muhakeme veya kod koruma gerekçesi sağlamaz. Erken kaynakların sonraki aşamada ortadan kalkması forgetting riskini ayrıca yaratır. Standart yeni adayda bütün retention aileleri eğitim boyunca erişilebilir olsun; öğrenme oranı evreleri ile veri ailesi sınırları birbirine bağlanmasın. Eski dört faz geçmiş koşu olarak korunabilir, yeni varsayılan olmak zorunda değildir.

Train/validation/test ayrımı belge hash'inden, paketleme ve karıştırmadan önce yapılmalı; aynı metnin farklı kaynakta kopyası farklı split'e geçmemelidir. Exact hash dedup near-duplicate temizliğinin kanıtı değildir. Küçük bir hash tabanlı holdout yeterlidir; corpusun %10'unu ayırmak gerekmez. Validation yaklaşık 5–10M token/language tavanıyla sınırlanabilir; bu bir başlangıç seçimi, uygulamanın mevcut sabit ayarı değildir. Benchmark/SFT eval metinleri için ayrıca contamination kontrolü gerekir. Kaynak lisansları ve kod verisi erişim şartları pinned source planından doğrulanmalıdır.

## Tokenizer: mevcut genişletmeyi koru, yeniden değiştirme

Yerel `customtokenizer/extension_report.json` 65.536 → 69.632 vocab ve 4.096 ek token/merge bildirir. Bu proje platformunun parçasıdır; yeni koşuda sessizce kaldırılmamalıdır. Ancak rapor, bütün eski ID/merge/özel-token invariants'ının ve kalite kazancının bağımsız kanıtı değildir.

Başlamadan eski bütün tokenların ID'lerinin aynı kaldığını, eski merge sırasının korunduğunu, normalizer/pretokenizer/decoder ve özel tokenların uyduğunu kontrol edin. Ek merge'ler İngilizce veya kod metinlerinde de yeni token üretebilir; append-only ID koruması aynı metnin aynı token dizisine dönüştüğünü garanti etmez. Türkçe/İngilizce/kod örneklerinde round-trip, karakter başına token ve yeni token kullanım oranlarını karşılaştırın. Yeni sembollerin öğrenilmemiş olması resmi model + genişletilmiş tokenizer baseline'ını doğrudan zayıflatabilir.

Resmi 65.536 vocab başlangıcında embedding resize gerekli; eski satırlar aynen korunmalı, yeni satırlar kontrollü başlatılmalı ve tied input/output embedding ilişkisi doğrulanmalıdır. Aynı 69.632 tokenizerla eğitilmiş temiz CPT'den resume ediliyorsa mevcut yeni satırları yeniden başlatmayın. Checkpoint tokenizerı başka ID düzeni taşıyorsa yalnızca boyutu eşitleyerek yüklemeyin; durun. Runtime/model/tokenizer revision ve dosya hash'leri cache/checkpoint manifestinde bulunmalıdır.

EACL 2026 tokenizer relearning araştırması yeni tokenizerların bazı dil uyarlamalarında faydalı olabileceğini gösterir, özellikle Latin dışı dillerde [S7]. Bu, mevcut Türkçe 4.096 ek tokenın optimum olduğu veya ikinci tokenizer operasyonunun şimdi maliyetini hak ettiği anlamına gelmez. Yeni genişletme ilk koşuya eklenmesin; kazanç fertility ve Türkçe task ölçümleriyle doğrulansın. Büyük bir embedding-only yeni araştırma hattı bu teslimin zorunlu parçası değildir.

## Tek GPU, token bütçesi ve öğrenme oranı

Tek A100 SXM 80 GB ilk adaydır; model bf16, full-weight CPT, optimizer/activation belleği gerçek preflight'ta doğrulansın. İlk aday sequence length 2.048; 1.024 ve 4.096 ile kısa karşılaştırma throughput, peak VRAM ve uzun Türkçe belge kapsamını belirlesin. Mamba/causal-conv kernel ve attention implementation'ın gerçek çalıştığı doğrulanmadan teorik hızdan maliyet çıkarmayın. Gradient checkpointing on/off ve micro-batch 1/2/4 seçeneklerinden sığanlar ölçülebilir. 8-bit optimizer full-weight eğitimle aynı şey değildir: tüm parametreler güncellenebilir, optimizer durumları düşük hassasiyetlidir. AdamW bf16/fp32 durumlarıyla karşılaştırmanın hız/bellek bedeli ölçülmeden seçim zorunlu sayılmasın.

Eski `micro_batch=1, seq_len=1024, accumulation=2048` yaklaşık 2,1M token/update demektir; 100M pilot yalnızca yaklaşık 48 optimizer update üretir. İlk aday 131.072–262.144 non-padding eğitim tokenı/update olsun; 100M pilotta yaklaşık 381–763 update verir. Daha büyük batch'in doğru veya yanlış olduğunu tek başına sayı belirlemez; fakat çok az optimizer update'li pilot LR davranışını yeterince göstermez. Kesin accumulation ölçülen micro-batch ve sequence length'ten hesaplanmalıdır.

Muhafazakâr başlangıç LR `2e-5`, gradient norm sınırı 1, küçük tek warmup ve sürekli schedule önerilir. Bunlar Jamba/Türkçe için deneyle doğrulanmış hiperparametreler değildir. Eski ilk faz `2e-4` değerinin on kat düşürülmesi, hazır post-training yeteneklerini korumak için risk azaltan başlangıç seçimidir; sonuç kötü olursa körlemesine daha uzun eğitim yapılmasın. Gerektiğinde kısa `1e-5`/`3e-5` karşılaştırması yapılabilir. LR warmup/decay ile replay'in CPT'de etkili olabileceğine dair birincil çalışma [S8] vardır; yayımlanan başka model ayarını birebir aktarmaz.

| Ölçek | Amaç | Karar |
| --- | --- | --- |
| 200–500 micro-step benchmark | VRAM, kernel, end-to-end token/s, resume | GPU preflight; kaliteli dil sonucu iddiası yok |
| İlk 100M eğitim tokenı | Aynı uzun koşunun pilot checkpoint'i | Baseline'dan loss ve yetenek değişimi, tekrar/karışım kontrolü |
| Toplam 1B eğitim tokenı | İlk uygulanabilir CPT tranche'ı | Pilot geçerse aynı optimizer/scheduler ile devam |
| 1B sonrası ek bütçe | Kazanç devam ediyorsa kapsam genişletme | Toplam 3B örnek üst aday; zorunlu tam-corpus epoch değil |

100M pilotu bağımsız 100M cosine endpoint'inde tamamen anneal edip aynı schedulerı yanlışlıkla yeniden başlatmayın. Başlangıçta 1B horizon seçilip 100M'de checkpoint/pause yapılarak aynı horizonla resume edilebilir; ya da warmup + sabit plateau kullanılıp bilinen endpoint'te ayrı açık decay branch'i seçilebilir. WSD bu esnek bütçe yaklaşımına araştırma desteği verir [S9]; repo cosine destekliyorsa ilk seçenek daha az geliştirme ister. Daha sonra horizon değişirse bunun schedule değişikliği olduğu kaydedilsin.

Bu repo koşu bütçesini attended input tokenı üzerinden uygular; shift sonrası loss-bearing token sayısı ayrıca raporlanır. SFT hedef bütçesi assistant loss-bearing tokenıdır; padding, tüketilmiş EOS/boundary, forward'a giren token ve shift sonrası prediction sayıları ayrıca kayıtlı olsun. Dataloader bittiğinde sessiz başa sarma uygun değildir. Manifest gerçek benzersiz belge/token kapsamını ve tekrar sayısını ayrı göstersin. 214.716.754 sınıflandırılmış belge, seçilen eğitim token sayısı değildir; label deposu metin içermediği için karakter/GB sayısından güvenilir token bütçesi çıkarılamaz.

Süre formülü: `training_hours = remaining_training_tokens / measured_end_to_end_tokens_per_second / 3600`. Buna baseline/eval, checkpoint upload ve veri hazırlama saatleri eklenir; bunlar ölçülen hıza dahilse iki kere eklenmez. Maliyet: `billable_gpu_hours × actual_hourly_quote + persistent_storage + transfer`. Sağlayıcı fiyatı/hızı burada ölçülmedi; altı gün zorunlu sınır veya yeni koşu süre tahmini değildir. İlk benchmark'ın warmup/compile süresi ayrı, steady-state ve checkpoint/eval dahil operasyonel hız ayrı raporlansın. Ortalama yanında yavaş shard/network davranışı da gösterilsin. Bir A10080'in 1B veya 3B tokenı kaç günde bitireceği henüz bilinmiyor.

## SFT kararı ve değerlendirme kapıları

CPT dil dağılımını öğretir; iyi ham Türkçe corpus otomatik instruction/JSON eğitim seti değildir. Asistan olarak kullanılacak model için CPT sonrası kısa ve yüksek kaliteli Türkçe + İngilizce SFT planlayın; CPT checkpoint'i ayrı korunsun. SFT'nin gerekli olup olmadığı baseline/pilot/final format eval ile doğrulansın. InstructionCP [S10], conversational kaybı azaltmak için template-aware CP alternatifi sunar; bunu tüm web metnini sahte asistan cevabına dönüştürmek veya garantili çözüm olarak uygulamayın. İlk varsayılan açık raw-text CPT + ayrı SFT'dir; gerekirse doğrulanmış instruction replay ayrı deney olarak eklenebilir.

SFT için yaklaşık 20–50M assistant target tokenı küçük başlangıç adayıdır; eski karttaki 40,96M nominal SFT tokenı bunun optimum olduğunu kanıtlamaz. Nominal packed-token ile loss-bearing assistant-token bütçeleri karıştırılmasın. Örnek sayısı sabitlenmeden içerik/uzunluk sayımı yapılsın. Çok miktarda düşük kaliteli sentetik satırdan önce doğrulanmış görev çeşitliliği, dedup ve doğru assistant-only mask kullanın. JSON/schema, extraction, kaynakla sınırlı QA, özetleme, çok turlu konuşma, kısa direkt talimat, TR/EN matematik/kod dahil olsun. Dataset kalite sınıflandırıcısının belge etiketleri bu SFT yanıtlarına dönüştürülmesin. Uygulamada chat template'in assistant mask desteği denetlensin [S11]. DPO/RL veya yeni milyarlarca instruction tokenı ilk teslim için zorunlu değildir.

Baseline, 100M, 1B ve CPT+SFT için aynı eval revision/prompt/decode ayarlarıyla şu tablo üretilecek:

| Boyut | Ayrı rapor |
| --- | --- |
| Türkçe dil modelleme | Source/content_type kırılımında heldout loss; aynı tokenizerla PPL |
| Türkçe görev içeriği | QA/özetleme/anlama doğruluğu; biçimden bağımsız değerlendirme |
| Talimat ve format | İlk deneme JSON parse, schema valid, doğru alan, ekstra metin, kısıt uyumu |
| Genel retention | İngilizce anlama ve talimat; matematik exact answer; kod test geçişi |
| Üretim bozulması | Tekrar döngüsü, anlamsız çıktı, yanlış dil, erken EOS/truncation oranı |
| Bağlam | 2K/8K/32K retrieval/grounding kontrolü; 256K korundu iddiası için ayrı test gerekir |

Tokenizerlar farklıysa token PPL doğrudan karşılaştırılmasın; ortak metinde byte/karakter normalize NLL veya görev puanı kullanın. MMLU seçeneğini log-likelihood ile ölçmek parsing karışıklığını azaltır; ama serbest generation instruction/format başarısını ayrıca ölçmek gerekir. JSON repair/retry/constrained decoding kullanılacaksa bunlar ayrı serving koşulu olarak raporlansın; ham modelin ilk-deneme başarısını şişirmesin. Çoklu tercih MMLU tek başına JSON güvenilirliğini ölçmez.

İlk küçük pilotta tüm görevler için istatistiksel başarı ilan etmeyin. Pratik pause örneği: iki eval penceresinde İngilizce heldout loss'un baseline'a göre %10'dan fazla kötüleşmesi veya format/task puanında 5 yüzde puan mutlak düşüş. Bunlar bilimsel evrensel eşik değil, ilk operasyonel alarmdır; küçük eval'de Wilson/bootstrap güven aralığı ve örnek incelemesiyle yorumlanmalıdır. NaN, hash uyuşmazlığı, tokenizer uyuşmazlığı, yanlış token oranı ve resume bozulması ise dil kalitesi yorumunu beklemeden teknik hata olarak durdurur. Türkçe loss kazanımı retention kaybını tek başına geçerli kılmaz.

## Repo kapsamı ve henüz bilinmeyenler

Repo; pinned source reconstruction + fail-closed hash join, route kararı manifesti, bounded streaming/cache, duplicate/split kaydı, token-temelli durma/karışım sayımı, full optimizer/scheduler/RNG/data cursor resume, baseline/eval entrypoint ve throughput/maliyet raporunu içermelidir. SFT ayrı config/data sözleşmesiyle eklenebilir. Yeni model mimarisi, distributed training, tüm ham korpusu VPS'e indirme ve büyük spec/test bürokrasisi bu küçük pahalı koşunun gereği değildir.

Henüz ölçülmeyenler: kaynak route/content_type kırılımı, kabul edilen gerçek token kapsamı, kaynaklar arası near-duplicate miktarı, Cosmos kabul güveni, temiz eski CPT revision'ı, mevcut tokenizer'ın round-trip/fertility/embedding audit'i, tek A10080 peak bellek/token hızı ve final karışımın retention etkisi. Bu belge GPU kiralamaz, job başlatmaz, eski sonuçları yeniden üretmiş sayılmaz. Yeni run fiyatı ve kapsamı bu preflight sonuçlarından çıkarılmalıdır.

## Birincil kaynaklar

- **S1 — AI21 Jamba2 eğitim açıklaması (2026):** https://www.ai21.com/blog/introducing-jamba2/ . Resmi mid-training/SFT/DPO/RL pipeline açıklaması; bu proje için ayrı eğitim sonucu değildir.
- **S2 — Kullanıcının mevcut model kartı:** https://huggingface.co/serda-dev/Jamba2-3B-Turkish . CPT/SFT kökeniyle ilgili mevcut beyan; gerçek ağırlık provenance'ı manifestten doğrulanmalı.
- **S3 — Resmi Jamba2-3B config ve kart:** https://huggingface.co/ai21labs/AI21-Jamba2-3B/raw/main/config.json ; https://huggingface.co/ai21labs/AI21-Jamba2-3B . Araştırmada `main` okundu; eğitim öncesi tam commit SHA pinlenmeli. Kartın bir tokenizer örneğinde farklı repo adı görünür; tek başına kopyalanmamalı.
- **S4 — Elhady, Agirre, Artetxe, ACL 2025, Emergent Abilities of Large Language Models under Continued Pre-training for Language Adaptation:** https://aclanthology.org/2025.acl-long.1547/ . Dil adaptasyonu/replay/downstream forgetting araştırması; Jamba2/Türkçe optimum oran kanıtı değildir.
- **S5 — Toukmaji, Flanigan, GEM 2025, Prompt, Translate, Fine-Tune, Re-Initialize, or Instruction-Tune?:** https://aclanthology.org/2025.gem-1.61/ . Geçerli çıktı/ICL kaybı değerlendirmesi; doğrudan Jamba benchmark'ı değildir.
- **S6 — Abbes ve ark., CoLLAs/PMLR 2026, Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models:** https://proceedings.mlr.press/v330/abbes26a.html . Llama ailesi ve çok dilli büyük ölçek replay araştırması; burada gradient-alignment ek altyapısı zorunlu önerilmez.
- **S7 — Jiang ve ark., EACL 2026, Tokenizer-Aware Cross-Lingual Adaptation of Decoder-Only LLMs through Embedding Relearning and Swapping:** https://aclanthology.org/2026.eacl-long.357/ . Tokenizer/embedding relearning alternatifi; mevcut append-only genişletmenin doğrulaması değildir.
- **S8 — Ibrahim ve ark., 2024, Simple and Scalable Strategies to Continually Pre-train Large Language Models:** https://arxiv.org/abs/2403.08763 ; HF indeksi https://huggingface.co/papers/2403.08763 . LR/replay yöntem motivasyonu; burada hiperparametre aktarımı yapılmaz.
- **S9 — Wen ve ark., ICLR 2025, Understanding Warmup-Stable-Decay Learning Rates:** https://proceedings.iclr.cc/paper_files/paper/2025/hash/6a1fe80a9e2dcda0b3e5fd0fd87eb097-Abstract-Conference.html ; https://arxiv.org/abs/2410.05192 . Esnek bütçeli schedule motivasyonu; Jamba2'de doğrudan doğrulanmamıştır.
- **S10 — Chen, Hwang, Lee, SIGTYP 2025, InstructionCP:** https://aclanthology.org/2025.sigtyp-1.1/ . Template-aware CP alternatifi; ilk koşuya otomatik taşınmaz.
- **S11 — Resmi runtime/SFT dokümantasyonu:** https://huggingface.co/docs/transformers/model_doc/jamba ; https://huggingface.co/docs/trl/sft_trainer . Runtime sürümü, kernel ve assistant loss-mask davranışı kullanılan pinned sürümde kontrol edilmelidir; genel Jamba dokümanındaki MoE örneği 3B config'inin yerine geçmez.
