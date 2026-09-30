# creddata

tolerance=0, x_policy=negative, strict_scope=labeled-files

| Tool | Prec (labeled) | Recall | F1 (labeled) | Prec (strict) | F1 (strict) | Hits | Time (s) |
|---|---|---|---|---|---|---|---|
| ours [BLOCK>=0.7] | 0.885 | 0.686 | 0.773 | 0.846 | 0.758 | 12887 | 106.1 |
| ours [WARN+BLOCK>=0.4] | 0.829 | 0.762 | 0.794 | 0.763 | 0.763 | 16005 | 106.1 |
| gitleaks | 0.861 | 0.458 | 0.598 | 0.836 | 0.592 | 8356 | 68.3 |
| trufflehog | 0.519 | 0.013 | 0.026 | 0.207 | 0.025 | 855 | 35.6 |
| detect-secrets | 0.528 | 0.258 | 0.347 | 0.286 | 0.271 | 15409 | 285.0 |


## ours [BLOCK>=0.7]

Secret recall by category (category, support, recall): [('Key', 3822, 0.704), ('Password', 2573, 0.836), ('UUID', 2316, 0.003), ('Secret', 1263, 0.959), ('PEM Private Key', 1161, 1.0), ('Auth:Basic Authorization', 601, 0.995), ('Token', 583, 0.873), ('URL Credentials', 213, 0.92), ('Auth', 174, 0.592), ('Token:UUID', 169, 0.994), ('AWS Client ID', 162, 0.37), ('Nonce', 148, 0.885), ('Key:Secret', 124, 0.952), ('API:Token', 112, 0.054), ('NTLM Token:Bearer Authorization:Auth', 105, 0.99)]

Non-secret flag rate by category (category, support, flagged): [('Key', 19929, 0.063), ('Password', 10941, 0.006), ('Token', 4594, 0.0), ('UUID', 3652, 0.0), ('API', 3598, 0.001), ('Auth', 3263, 0.002), ('Secret', 2175, 0.001), ('Credential', 547, 0.004), ('Basic Authorization', 538, 0.0), ('URL Credentials', 288, 0.007), ('Auth:Token', 268, 0.0), ('Key:Token', 263, 0.004), ('API:Key', 201, 0.0), ('Key:Secret', 145, 0.007), ('Password:URL Credentials', 117, 0.0)]

## ours [WARN+BLOCK>=0.4]

Secret recall by category (category, support, recall): [('Key', 3822, 0.92), ('Password', 2573, 0.895), ('UUID', 2316, 0.006), ('Secret', 1263, 0.978), ('PEM Private Key', 1161, 1.0), ('Auth:Basic Authorization', 601, 0.995), ('Token', 583, 0.914), ('URL Credentials', 213, 0.972), ('Auth', 174, 0.684), ('Token:UUID', 169, 0.994), ('AWS Client ID', 162, 0.37), ('Nonce', 148, 0.973), ('Key:Secret', 124, 0.96), ('API:Token', 112, 0.098), ('NTLM Token:Bearer Authorization:Auth', 105, 0.99)]

Non-secret flag rate by category (category, support, flagged): [('Key', 19929, 0.109), ('Password', 10941, 0.018), ('Token', 4594, 0.002), ('UUID', 3652, 0.0), ('API', 3598, 0.006), ('Auth', 3263, 0.003), ('Secret', 2175, 0.004), ('Credential', 547, 0.005), ('Basic Authorization', 538, 0.0), ('URL Credentials', 288, 0.042), ('Auth:Token', 268, 0.004), ('Key:Token', 263, 0.011), ('API:Key', 201, 0.01), ('Key:Secret', 145, 0.014), ('Password:URL Credentials', 117, 0.017)]

## gitleaks

Secret recall by category (category, support, recall): [('Key', 3822, 0.881), ('Password', 2573, 0.058), ('UUID', 2316, 0.004), ('Secret', 1263, 0.835), ('PEM Private Key', 1161, 1.0), ('Auth:Basic Authorization', 601, 0.0), ('Token', 583, 0.518), ('URL Credentials', 213, 0.0), ('Auth', 174, 0.207), ('Token:UUID', 169, 0.923), ('AWS Client ID', 162, 0.698), ('Nonce', 148, 0.0), ('Key:Secret', 124, 0.629), ('API:Token', 112, 0.045), ('NTLM Token:Bearer Authorization:Auth', 105, 0.0)]

Non-secret flag rate by category (category, support, flagged): [('Key', 19929, 0.049), ('Password', 10941, 0.002), ('Token', 4594, 0.005), ('UUID', 3652, 0.0), ('API', 3598, 0.004), ('Auth', 3263, 0.005), ('Secret', 2175, 0.005), ('Credential', 547, 0.007), ('Basic Authorization', 538, 0.0), ('URL Credentials', 288, 0.0), ('Auth:Token', 268, 0.0), ('Key:Token', 263, 0.0), ('API:Key', 201, 0.01), ('Key:Secret', 145, 0.014), ('Password:URL Credentials', 117, 0.0)]

## trufflehog

Secret recall by category (category, support, recall): [('Key', 3822, 0.001), ('Password', 2573, 0.003), ('UUID', 2316, 0.0), ('Secret', 1263, 0.013), ('PEM Private Key', 1161, 0.012), ('Auth:Basic Authorization', 601, 0.0), ('Token', 583, 0.031), ('URL Credentials', 213, 0.202), ('Auth', 174, 0.006), ('Token:UUID', 169, 0.006), ('AWS Client ID', 162, 0.049), ('Nonce', 148, 0.0), ('Key:Secret', 124, 0.0), ('API:Token', 112, 0.0), ('NTLM Token:Bearer Authorization:Auth', 105, 0.0)]

Non-secret flag rate by category (category, support, flagged): [('Key', 19929, 0.001), ('Password', 10941, 0.002), ('Token', 4594, 0.0), ('UUID', 3652, 0.0), ('API', 3598, 0.0), ('Auth', 3263, 0.001), ('Secret', 2175, 0.0), ('Credential', 547, 0.0), ('Basic Authorization', 538, 0.002), ('URL Credentials', 288, 0.292), ('Auth:Token', 268, 0.0), ('Key:Token', 263, 0.0), ('API:Key', 201, 0.0), ('Key:Secret', 145, 0.0), ('Password:URL Credentials', 117, 0.256)]

## detect-secrets

Secret recall by category (category, support, recall): [('Key', 3822, 0.033), ('Password', 2573, 0.66), ('UUID', 2316, 0.016), ('Secret', 1263, 0.225), ('PEM Private Key', 1161, 0.274), ('Auth:Basic Authorization', 601, 0.017), ('Token', 583, 0.334), ('URL Credentials', 213, 0.559), ('Auth', 174, 0.144), ('Token:UUID', 169, 0.0), ('AWS Client ID', 162, 0.698), ('Nonce', 148, 0.493), ('Key:Secret', 124, 0.645), ('API:Token', 112, 0.054), ('NTLM Token:Bearer Authorization:Auth', 105, 0.0)]

Non-secret flag rate by category (category, support, flagged): [('Key', 19929, 0.014), ('Password', 10941, 0.224), ('Token', 4594, 0.003), ('UUID', 3652, 0.0), ('API', 3598, 0.02), ('Auth', 3263, 0.014), ('Secret', 2175, 0.232), ('Credential', 547, 0.004), ('Basic Authorization', 538, 0.006), ('URL Credentials', 288, 0.444), ('Auth:Token', 268, 0.0), ('Key:Token', 263, 0.034), ('API:Key', 201, 0.488), ('Key:Secret', 145, 0.51), ('Password:URL Credentials', 117, 0.444)]
