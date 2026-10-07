# Yaklaşık 97 dolarla Jamba2-3B Türkçe CPT planı

7 Ekim 2026. Bu belge araştırma ve bütçe planıdır; yeni eğitim başlatılmadı. GPU hızları [pilot raporundan](CPT_GPU_PILOT_2026-10-07.md), veri karşılaştırması gerçek Hub etiketlerinden ve yerel classifier denetiminden gelir.

## Karar ve çalışma hedefi

İlk gerçek koşu **toplam 1 milyar attended input token**, bunun yaklaşık **800 milyonu Türkçe** olsun. Tek A100 üzerinde bütün ağırlıklar güncellensin. Hedef Türkçe ifade, terim kullanımı ve metin modelleme/anlama uyarlamasıdır. Büyük corpusun tamamını tüketmek, bütün uzmanlık alanlarında yeni bilgi öğretmek, uzun bağlam eğitimi ve kapsamlı muhakeme/talimat iyileştirmesi bu koşunun başarı ölçütü değildir. Kalite kazanımı henüz ölçülmedi; 1B token bilimsel olarak doğrulanmış bir yeterlilik eşiği değildir.

Önceki 2–3B token önerisine göre kapsam azalır. Mevcut teknik config zaten 1B hedefliyordu; öneri bu hedefi bütçeli ilk CPT teslimi olarak somutlaştırmaktır. 100M kontrolü ayrı bir eğitim değildir: aynı 1B schedule ve optimizer ile koşunun ilk kısmıdır. Kontrol geçerse sıfırdan yeniden başlayıp bu tokenları tekrar ödemeden devam edilir.

## Model ve Jamba mimarisinin etkisi

Başlangıç `ai21labs/AI21-Jamba2-3B`, revision `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9`. Resmi config 28 katman, attention period 14/offset 7 ve tek expert bildirir: bu 3B sürüm yoğun bir Mamba/attention hibritidir; genel Jamba MoE toplam/aktif parametre sayılarını bu koşuya taşımayız [S1]. Model kartı resmi ağırlığın mid-training, SFT, DPO ve RL sonrası yayımlandığını anlatır [S2]. Türkçe CPT mevcut talimat yeteneğini değiştirebilir.

SSM yapısının uzun bağlam çıkarım avantajı, CPT'nin belirli bir token bütçesiyle yeterli öğrenme sağlayacağını veya her GPU'da teorik TFLOPS oranında hızlanacağını göstermez. Bizim dayanağımız aynı modelin ölçülen A100 hızıdır. Bu bütçede 2K eğitim bağlamı korunur; 256K kapasitenin korunduğu veya geliştirilmiş olduğu iddia edilmez. Mevcut 69.632-token tokenizer ve tüm ağırlıkların eğitimi korunur; ek vocab tekrar değiştirilmez.

## Veri karşılaştırması

Hub etiket revision'ı `388300940d3675640553e817e2b6eae1b3b6947b`; 2.151 Parquet parçası, toplam 214.716.754 sınıflandırılmış kayıt doğrulandı. Bunlar metin değil, metnin pinned kaynaktan yeniden eşleştirilmesini sağlayan metadata'dır.

24 parçanın ilk/orta/son konumlarından eşit aralıklı toplam **234.129 etiket** incelendi. Aşağıdaki kabul oranı `PREMIUM` veya `KEEP` ve `input_truncated=false` koşuludur; ham metin, dedup ve tokenize kontrollerinden sonra miktar ayrıca azalabilir. Örnekleme corpus-geneli rastgele veya token-ağırlıklı bir tahmin değildir. Etiketlerdeki content type da öğrenci tahminidir. [Örnekleme kanıtı](evidence/CPT_LABEL_SAMPLE_2026-10-07.json).

| Kaynak | Tamamlanan sınıflandırma | Örnekte kabul | Bu koşudaki rol |
| --- | ---: | ---: | --- |
| FineWeb2 Türkçe | 95.129.129 | %49,3 | Genel web; kabul örneğinin yaklaşık %60'ı haber etiketli |
| Mogan | 32.462.870 | %56,9 | Genel web; kabul örneğinin yaklaşık %64'ü haber etiketli |
| Bella Akademik | 965.862 | %80,0 | Terminoloji/açıklayıcı metin; kabul örneğinin yaklaşık %97'si akademik etiketli |
| Bella Özenli | 1.588.070 | %63,9 | Web kaynaklarına göre daha çeşitli üslup ve konular |
| Bella temiz-mC4 | 55.882.753 | %67,8 | Ek web çeşitliliği; haber tekrarı nedeniyle küçük pay |
| Bella temiz-OSCAR | 19.612.617 | %60,4 | Ek web çeşitliliği; küçük pay |
| Cosmos | 9.075.453 | %11,0 | İlk koşuya dahil edilmesin |

Yerel `routing-audit.json` 3.968 Teacher holdout kaydı içeriyor. Denetlenen model dosyasının SHA-256'sı Hub üretici kimliğiyle eşleşiyor. Öğrencinin `KEEP/PREMIUM` seçtiklerinin Teacher tarafından da bu iki sınıfa kabul edilme oranı FineWeb %95,6, Mogan %96,4, Bella %95,6, Cosmos %82,4. Altı rotanın birebir uyumu sırasıyla %81,3/%85,0/%85,2/%61,9. Bunlar aynı dahili holdout üzerindeki Teacher uyumudur; insan doğruluğu veya tüm corpusun temizliği değildir. Teacher'ın onarım/review gerektirdiği bazı metinler öğrencide kabul edilmiştir. Cosmos'un düşük uyumu ve düşük kabul verimi, sınırlı ilk bütçede onu dışarıda tutmayı destekliyor. [Denetim ve teklif kanıtı](evidence/CPT_BUDGET_EVIDENCE_2026-10-07.json).

Kaynak kartları Bella Akademik'i yayın/tez, Özenli'yi seçilmiş web içeriği olarak tanımlar; Mogan kendi içinde near-duplicate temizliği bildirir [S5, S6]. Bu, farklı corpuslar arasında near-duplicate olmadığı anlamına gelmez. Mevcut adapter global exact-hash dedup yapıyor. Bella kartındaki eski satır/byte sayıları bazı gerçek DONE kayıtlarından farklıdır; tamamlanan sınıflandırma sayılarında DONE metadata kullanıldı.

## Eğitimde görülecek token dağılımı

| Kaynak | Toplam koşudaki token | Toplam pay |
| --- | ---: | ---: |
| FineWeb2 Türkçe | 200M | %20 |
| Mogan | 200M | %20 |
| Bella Akademik | 160M | %16 |
| Bella Özenli | 160M | %16 |
| Bella temiz-mC4 | 40M | %4 |
| Bella temiz-OSCAR | 40M | %4 |
| FineWeb-Edu İngilizce | 140M | %14 |
| OpenWebMath | 40M | %4 |
| StarCoder Python | 20M | %2 |

Bu başlangıç dağılımı bir mühendislik seçimidir. Türkçe payının %40'ı akademik/özenli kaynaklara ayrılarak web/haber ağırlığı azaltılır. Küçük İngilizce replay dil uyarlamasında downstream yetenek kaybını azaltabilmektedir [S3, S4]; %20'nin Jamba/Türkçe için optimum olduğu kanıtlanmış değildir. Kaynak karışımı bütün koşu boyunca sürsün; ayrı web/math/code fazlarına bölünmesin.

Türkçe kaynak oranları **token hedefidir**. Mevcut `source_weights` ham belge interleaving ağırlığıdır ve bu tabloyu tek başına garanti etmez. Küçük kabul edilmiş metin örneğinin gerçek tokenizer çıktısıyla ağırlıklar ayarlanmalı; hazır cache'in token sayaçlarıyla gerçek dağılım raporlanmalıdır. Kaynak oranı uygulanmış gibi gösterilmemeli.

İlk etiket havuzu için FineWeb ilk 1,5M, Mogan ilk 1,5M, Akademik mevcut tüm 965.862, Özenli ilk 1,5M, mC4 ve OSCAR ilk 300k'şer kayıt yaklaşık 6,07M metadata satırı eder. Bunlar eğitim token miktarı garantisi değildir; gerçek kabul edilmiş token verimi cache hazırlığında ölçülür. `max_labels` 7M ve audit üst sınırı 10 GiB adaydır. Mevcut 2M varsayılan sınırı bu seçimi karşılamaz. İlk contiguous aralıklar pahalı corpus taramasını azaltır, fakat sıra yanlılığı taşır; elde edilen model corpus-geneli temsil iddiasıyla sunulmaz. Yeni doc ID/hash şeması icat edilmez.

`input_truncated` classifier'ın baş/son örneklemesini bildirir; kaynak metnin fiziksel kesildiğini söylemez. İlk koşudaki mevcut ihtiyatlı dışlama uzun metin kapsamını azaltıyor. Bütçeyi doldurmak için bu kural sessizce gevşetilmesin. Kabul edilen metinler yeniden kaynaktan alınarak tam UTF-8 SHA-256/doc ID kontrolü ve hash holdout yapılır; mümkün olduğunca benzersiz belgeler kullanılır, `repeat_dataset=false` korunur.

## Eğitim ayarları ve kontrol

Pilotla aynı bf16 full-weight causal LM, 2048 sequence, microbatch 2, accumulation 32, 131.072 token/update, AdamW8bit ve gradient checkpointing kullanılsın. LR `2e-5`, %2 warmup ve tek 1B cosine horizon başlangıç adayıdır. Yaklaşık 7.630 optimizer update elde edilir. Batch/epoch'u büyütmek için bütçe harcanmaz; seçilen cache üzerinde sonlu token bütçesi uygulanır.

Baseline, yaklaşık 100M ve finalde aynı sabit Türkçe/İngilizce holdout, görev ve JSON/talimat testleri çalışsın. 100M noktasındaki yaklaşık 763 update toplam bütçeye dahildir. Türkçe heldout kaybının azalmasıyla birlikte retention ve görev sonuçları kontrol edilir; yalnızca train loss'a bakılmaz. Mevcut 12-case diagnostic geniş yetenek kanıtı sayılmaz. Beklenen çıktı ölçümleriyle saklanan CPT checkpoint'idir; bu bütçe yeni SFT eğitimini kapsamıyor.

## Bütçe ve süre

LinguAI team kredisi kontrol sırasında **$97,35** idi. Güncel on-demand A100 SXM4 80GB teklif `49739779`: 180 GiB disk dahil **$1,23/saat**, 16 effective CPU core, yaklaşık 256 GB RAM; indirme $0,004/GB, yükleme $0,00533/GB. Teklif değişebilir. Pilot aynı sınıf A100'de 5.483,5 token/sn ölçtü; yeni hostta aynı hız garanti değildir.

| Eğitim tokenı | Pilot hızıyla saat / ücret | 4.500 token/sn senaryosunda saat / ücret |
| --- | ---: | ---: |
| 750M | 38,0 h / $46,73 | 46,3 h / $56,94 |
| **1B hedef** | **50,7 h / $62,31** | **61,7 h / $75,93** |
| 1,2B | 60,8 h / $74,77 | 74,1 h / $91,11 |

Son sütun checkpoint dahil ölçülmüş hız değil, yavaşlama senaryosudur. İlk hedef 1B kalsın: 1,2B yavaş hostta ek giderler için yeterli pay bırakmıyor.

Veri/hash join ve tokenize hazırlığı **GPU kiralanmadan VPS CPU'sunda** yapılır. 1B cache mevcut int32 token + uint8 token provenance düzeninde yaklaşık 5 GB'dır; audit indeksi ve geçici dosyalar ayrıca yer tutar. Tüm ham corpus VPS'e indirilmez. Görev diski kullanımı sınırlandırılır; hazırlıkta en az 15 GB VPS boş alanı korunur. GPU'nun 180 GiB diski model/cache, optimizer checkpoint'leri ve atomik kayıt geçişi için pay sağlar; nihai boyut ilk gerçek checkpoint'te ölçülür.

Plan: eğitim $62–76, toplam 4–6 GPU saatlik indirme/başlangıç/değerlendirme/kayıt payı $5–7,5 ve transfer için $3 rezerv. **Toplam yaklaşık $75–90 ve 2,5–3 gün** hedeflenir; bu bir teklif/performans senaryosudur. $97'nin tamamı hazırlığa harcanmaz. Yeni benchmark turları önerilmiyor.

Checkpoint her yaklaşık 500 adımda (~65,5M token, pilot hızında ~3,3 saat), en fazla iki normal yerel checkpoint; 100M kontrol ve final ayrıca saklanır. Latest/best/final uygun anlarda Hub'a doğrulanarak aktarılır; her ara checkpoint'in süresiz biriktirilmesi gerekmez. Instance silinmeden son ağırlık, tokenizer, manifest, eval ve resume durumu off-host doğrulanır.

Paid koşu başladıktan sonra instance fiyatı × geçen süre + transfer üzerinden tüm işi kapsayan harcama takip edilir. $90'da yeni harcama için devam kararı alınmaz; checkpoint/yükleme için ayrılmış payla kapanışa geçilir. **$92 toplam iş bütçesi** hedef durdurma sınırıdır, kalan kredi tampon kalır. Bu sınır şu an trainer içinde uygulanmış değildir; kira öncesinde dış watchdog eklenmelidir. 1B'ye ulaşmadan bütçe/kalite sınırı gelirse token sayısı ve `stopped` durumu doğru raporlanır; tam 1B tamamlandı denmez ve schedule sessizce yeniden yazılmaz.

## Kira öncesindeki sınırlı işler

1. VPS'de seçili kaynak/etiket aralıklarıyla gerçek 1B cache ve ayrı küçük validation cache oluştur; yukarıdaki token dağılımını, hashleri, disk kullanımını ve temiz üretici çıkışını kontrol et. Pilot hazırlayıcısının geçerli manifest sonrası exit 139 vermesi hâlâ açık; bu çözülmeden otomatik tam koşu açma.
2. Checkpoint upload doğrulaması ve $92 harcama watchdog'unu hazırla. Bunlar ek ücretli GPU benchmark'ı gerektirmez.
3. Cache hazırken tek A100 kirala; baseline + aynı eğitimin 100M kontrolü + 1B devamı. İlk gerçek kayıtta resume/storage maliyetini ve operasyonel hızı ölçerek kalan bütçeyi güncelle.

## Kaynaklar

- S1: [AI21 Jamba2-3B config](https://huggingface.co/ai21labs/AI21-Jamba2-3B/raw/525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9/config.json).
- S2: [AI21 resmi model kartı](https://huggingface.co/ai21labs/AI21-Jamba2-3B).
- S3: [Elhady ve ark., ACL 2025: dil uyarlamasında İngilizce replay ve downstream forgetting](https://aclanthology.org/2025.acl-long.1547/). Deneyleri Jamba/Türkçe değil; optimum 1B bütçe kanıtı olarak kullanılmadı.
- S4: [Ibrahim ve ark.: Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763). LR/replay motivasyonu; başka modelin hiperparametreleri birebir aktarılmadı.
- S5: [Pinned BellaTurca kartı](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca/blob/05097c1469d4c96780bdd18ec64c7948d9ecf771/README.md).
- S6: [Pinned Mogan kartı](https://huggingface.co/datasets/moganai/mogan-turkish-web/blob/4f1337fed9cdc6d6c0cc7bcc59fb1bafd491c0d0/README.md).
