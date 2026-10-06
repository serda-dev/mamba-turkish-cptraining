# CPT projesi: mevcut durum ve hedef

Bu belge, yeni CPT çalışmasını hazırlayacak agent için kısa görev bağlamıdır. Eğitim mimarisi ve hiperparametreleri burada kesinleştirmiyoruz; bunları proje kodunu ve ölçümleri inceleyerek sen belirle.

## Şimdiye kadar ne yaptık?

- Daha önce `ai21labs/AI21-Jamba2-3B` ağırlıklarından başlayarak dört fazlı Türkçe CPT yaptık. Yaklaşık 48 GB VRAM'li tek GPU'da koşu yaklaşık altı gün sürdü. Önceki veri karışımında İngilizce oranı yüksekti; Türkçe verinin kalite filtresi bugünkünden daha zayıftı. Bu süreyi yeni koşu için zorunlu sınır veya tüm veri üzerinde bir epoch kanıtı sayma.
- Sonrasında yaklaşık 40 bin örneği Teacher modelle etiketleyip küçük bir kalite sınıflandırıcısı eğittik. Bu sınıflandırıcıyla dört Türkçe dataset ailesinden **214.716.754 belgeyi** sınıflandırdık. Sonuçlar özel Hugging Face deposu [`serda-dev/filtered-turkish-dataset`](https://huggingface.co/datasets/serda-dev/filtered-turkish-dataset) içinde 2.151 Parquet parçası olarak doğrulandı. Sınıflandırma GPU instance'ı iş bitince silindi.
- Kaynak aileleri: `HuggingFaceFW/fineweb-2` (`tur_Latn`), `turkish-nlp-suite/BellaTurca` (dört alt küme), `ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0` ve `moganai/mogan-turkish-web`. Sabitlenmiş revision, config, split, metin alanı ve lisans bilgileri `serda-dev/linguai-dataset-quality` projesindeki `deploy/source_plan_2026-10-02.json` dosyasında.
- Hugging Face çıktı deposu **metinleri içermiyor**. Her satırda `source_id`, `row_ordinal`, `doc_id`, `document_hash`, `input_bytes`, `input_truncated`, `predicted_label_json` ve `predicted_route` var. Eğitim metni, sabitlenmiş kaynak sürümünden yeniden okunup kimlik ve hash ile doğrulanmalı. VPS'de bütün ham korpusu tutmak istemiyoruz.

## Amacımız

Jamba2 3B için, önceki çalışmadan **daha kaliteli ve Türkçe ağırlığı daha yüksek** bir CPT koşusu hazırlamak. Tam ağırlık CPT ve modelin öğrenme kapasitesi korunacak. İngilizce, matematik ve kod içeriğini genel yetenek kaybını önleyecek ölçüde eklemek istiyoruz; kesin oranı ve faz tasarımını sen belirleyip gerekçelendir. Dört fazı korumak mümkündür, fakat eski fazları ve eski GB hedeflerini otomatik olarak kopyalama.

Şimdiki donanım tercihi **tek GPU**. 48 GB VRAM önceki koşuda sıkışıktı; ilk aday 80 GB A100 SXM. Birden fazla küçük GPU'yu birleştiren çözüm veya dağıtık eğitim şu an istenmiyor. Önceki altı günlük süre bir karşılaştırma noktasıdır, sabit eğitim süresi sınırı değildir. Eğitim kapsamını seçilmiş verinin gerçek token sayısı ve ölçülen hız üzerinden açıkça tanımla.

## Projede gereken işler

1. Etiket deposunu CPT için kullanılabilir hale getir: kaynak metni sabitlenmiş revision'dan akışla oku, `source_id` + `row_ordinal` üzerinden eşleştir, `document_hash` ve `doc_id` doğrulamasını yap. Uyumsuzlukta dur. Kaynaklar arası tekrarları önle. Etiketlerin hiçbiri kaybolmadan hangi belgelerin eğitime girdiğini izlenebilir kıl.
2. Başlangıç seçimi olarak `PREMIUM` ve `KEEP` belgelerini değerlendir. `REVIEW`, `DETERMINISTIC_REPAIR`, `MODEL_REPAIR` ve `DROP` için ayrı karar ver; onarım uygulanmadan onarım gerektiren metni doğrudan temiz metin sayma. Cosmos kaynağının kabul kararları diğer kaynaklara göre daha zayıf doğrulandığı için onu ayrıca kontrol et.
3. Bu branch'in mevcut veri girişi eski `MRBeDev/MC4veOSCARTekrarsizBirlestirilmis` JSONL/GZ düzenine bağlı. Yeni, etiketlerle eşleşmiş metinleri fazlara ve token cache'e taşıyan yolu ekle. Tam veri VPS diskinde birikmemeli; ara çıktılar ve checkpoint'ler güvenli biçimde saklanıp yerel disk temizlenebilmeli.
4. Mevcut konfigürasyonda fazlar `english_sources` kullanıyor, fakat varsayılan `iter_english_texts()` yolu `english_dataset` okuyor. Eski koşunun gerçek karışımını varsayma; yeni koşuda Türkçe/İngilizce token oranını doğrula ve raporla.
5. Eğitim döngüsü şu anda `max_steps` veya süre sınırıyla duruyor ve veri bittiğinde başa sarıyor. Yeni koşu için **işlenen token**, benzersiz belge kapsamı, tekrar edilen token ve faz başına durma koşulunu açıkça kaydet. Checkpoint/resume ve değerlendirme çalışmalı. Hangi ağırlıktan devam edileceğini doğrula: önceki temiz CPT checkpoint'i varsa kullanılabilir; SFT uygulanmış modeli farkında olmadan CPT başlangıcı yapma. Mevcut branch'in varsayılanı pretrained `ai21labs/AI21-Jamba2-3B`.

## Beklenen teslim

Veri yolunun doğrulandığı, tek GPU'da başlatılıp kaldığı yerden sürdürülebilen CPT akışı; hangi veri ve kaç tokenın işlendiğini gösteren manifest/ölçümler; eğitim öncesi kısa test ve gerçek hıza dayalı süre/maliyet tahmini. Detaylı eğitim planını ve teknik seçimleri bu bağlamdan hareketle sen oluştur.
