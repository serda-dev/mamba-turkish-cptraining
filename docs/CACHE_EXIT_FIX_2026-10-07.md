# Cache hazırlama çıkış hatasının düzeltilmesi

## Hata ve düzeltme

Eski `serdadev/jamba2-cpt:pilot-786a53b-r2` image'ında 10.000 tokenlık gerçek kaynak cache'i başarıyla yazıldıktan sonra süreç **exit 139** ile çöküyordu:

```text
Fatal Python error: PyGILState_Release: thread state ... must be current when releasing
Python runtime state: finalizing
```

Sorunlu birleşim Python 3.11.10, `datasets 5.0.1`, `PyArrow 25.0.1` idi. Python iterator'larının kapanması tek başına bu native-reader kapanış hatasını çözmedi. Arrow/HF tarafındaki Parquet fragment scanner için aynı hata sınıfı [upstream kayıtta](https://github.com/apache/arrow/issues/45214) anlatılıyor. Bir başka makinedeki bütün sürümlerin aynı şekilde bozuk olduğunu iddia etmiyoruz.

Doğrulanan çözüm `datasets==3.1.0`, `pyarrow==19.0.1`, `huggingface_hub==0.36.2`, `fsspec==2024.9.0` birleşimini CPU gereksinimlerinde ve GPU kurulumunda sabitlemek. Bu datasets sürümü kullandığımız Parquet akışında fragment scanner yerine `ParquetFile.iter_batches()` yolunu kullanır. Ayrıca token limiti veya hata geldiğinde packer → mixer → dil kaynağı → HF/raw okuyucu zincirinin tamamı açıkça kapatılır. Bir okuyucu kapanışı hata verirse cache `complete=true` olarak yayımlanmaz.

`os._exit`, hata kodunu değiştirme, rastgele kapanış beklemesi veya manifesti görüp 139'u başarı sayma kullanılmaz. Tokenizer, veri kimliği ve cache dosya formatı değişmez. GPU model/optimizer/CUDA kernel ayarları değiştirilmedi; bunun için yeniden GPU kiralanmadı.

## Doğrulama

- Eski ortamda aynı 10k komutu exit 139 ile tekrarlandı.
- Sabitlenmiş ortamda aynı komut exit 0 ile tamamlandı; token sayısı, kaynak sayaçları ve dört tür shard dosyasının SHA-256'ları eski geçerli çıktıyla birebir eşleşti.
- 45 ilgili test sabitlenmiş Docker ortamında geçti. Eklenen kontroller erken token limiti geldiğinde bütün dil kaynaklarının kapanmasını ve kapanış hatasında tamamlanmış manifest yazılmamasını sınar.
- Altı Türkçe kaynak + üç İngilizce/math/code replay kaynağı ile **2.000.000 gerçek token** üretildi, süreç exit 0 ile kapandı. 600.000 etiket indekslendi, 7.309 ham kaynak satırının doc ID/hash'i doğrulandı ve bütün altı Türkçe kaynaktan kabul edilmiş metin görüldü. Cache 1.592.970 Türkçe ve 407.030 İngilizce kaynak tokenı içeriyor; bütün shard bütünlük kontrolleri geçti. Join denetimi token limitinde durduğu için corpus-geneli `complete` iddiası taşımıyor. [Ölçüm kanıtı](evidence/CACHE_EXIT_FIX_2026-10-07.json).

Test logları VPS'de `/home/serda/cpt-cache-fix-check/` altındadır. Üretim image'ı `serdadev/jamba2-cpt:pilot-786a53b-r3`; eski r2 üzerinde sadece config değiştirerek hata çözülmüş sayılmaz. Yeni bağımlılık yükseltmelerinde erken token duruşu sonrası **süreç çıkışı ve cache bütünlüğü birlikte** yeniden kontrol edilmelidir.

Bu düzeltme 1B tokenın tamamını hazırladığımız veya kalite değerlendirmeli CPT'yi başlattığımız anlamına gelmez. Büyük cache, gerçek kaynak-token dağılımı, checkpoint upload ve bütçe watchdog hazırlığı ayrıca tamamlanır.
