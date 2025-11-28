# 🎨 UI Mockups Reference

This document provides detailed descriptions and implementation notes for each UI mockup design.

---

## 📁 Mockup Files

All mockup images are located in `design/mockups/`:

1. `01_hero_section.png` - Landing page hero section
2. `02_prediction_form.png` - Prediction input form
3. `03_results_card.png` - Results display with recommendations
4. `04_performance_dashboard.png` - Model performance metrics
5. `05_features_section.png` - Feature cards grid

---

## 1️⃣ Hero Section (`01_hero_section.png`)

### Description
The hero section is the first thing users see when landing on the homepage. It sets the tone for the entire application with a premium, modern design.

### Key Elements

#### Background
- **Base Color**: Deep dark navy (`hsl(220, 15%, 8%)`)
- **Gradient Overlay**: Vibrant green to blue gradient (`hsl(142, 71%, 45%)` → `hsl(210, 100%, 56%)`)
- **Decorative Elements**: Floating agricultural icons (leaf, water drop, sun) with subtle animation
- **Effect**: Parallax scrolling on scroll

#### Typography
- **Main Heading**: "Grow Smarter with AI-Powered Crop Recommendations"
  - Font: Outfit (Display font)
  - Size: `clamp(2.5rem, 2rem + 2.5vw, 4rem)`
  - Weight: 800 (Extrabold)
  - Effect: Gradient text with subtle animation
  - Color: White with gradient overlay

- **Subheading**: "Analyze soil nutrients and environmental conditions to get instant, data-driven crop suggestions with 99.3% accuracy"
  - Font: Inter
  - Size: `clamp(1.125rem, 1rem + 0.625vw, 1.25rem)`
  - Weight: 400 (Normal)
  - Color: Light gray (`hsl(0, 0%, 70%)`)

#### Call-to-Action Buttons
1. **Primary Button**: "Get Crop Recommendation"
   - Background: Green gradient with glow
   - Hover: Lift effect with enhanced glow
   - Click: Ripple animation

2. **Secondary Button**: "View Model Performance"
   - Background: Glassmorphism effect
   - Border: Subtle white border
   - Hover: Border color changes to green with glow

#### Stats Cards (Bottom)
Four floating glassmorphism cards displaying:
- **99.3% Accuracy** - Target icon
- **22 Crop Types** - Seedling icon
- **8.5ms Speed** - Lightning icon
- **200+ Tests** - Checkmark icon

**Card Styling**:
- Background: Frosted glass with backdrop blur
- Border: Subtle white border (`hsla(0, 0%, 100%, 0.1)`)
- Shadow: Soft shadow with glow
- Animation: Floating effect (subtle up/down motion)

### Implementation Notes

```html
<section class="hero">
  <div class="hero-background">
    <!-- Particle system or animated background -->
  </div>
  <div class="hero-content container">
    <h1 class="hero-title text-gradient">
      Grow Smarter with AI-Powered Crop Recommendations
    </h1>
    <p class="hero-subtitle">
      Analyze soil nutrients and environmental conditions to get instant, 
      data-driven crop suggestions with 99.3% accuracy
    </p>
    <div class="hero-cta">
      <a href="/predict" class="btn btn-primary">Get Crop Recommendation</a>
      <a href="/performance" class="btn btn-secondary">View Model Performance</a>
    </div>
  </div>
  <div class="hero-stats container">
    <div class="stat-card card-glass">
      <div class="stat-value">99.3%</div>
      <div class="stat-label">Accuracy</div>
    </div>
    <!-- Repeat for other stats -->
  </div>
</section>
```

---

## 2️⃣ Prediction Form (`02_prediction_form.png`)

### Description
A centered glassmorphism card containing the input form for crop prediction. Clean, modern, and user-friendly with real-time validation.

### Key Elements

#### Card Container
- **Background**: Glassmorphism with frosted glass effect
- **Border**: Subtle gradient border (appears on hover)
- **Border Radius**: `1rem` (--radius-xl)
- **Padding**: `2rem` (--space-6)
- **Max Width**: 800px
- **Shadow**: Soft shadow with subtle glow

#### Form Header
- **Icon**: 🌾 emoji or crop icon
- **Title**: "Crop Recommendation Input"
- **Font**: Outfit, size `clamp(1.5rem, 1.3rem + 1vw, 2rem)`
- **Color**: White

#### Form Sections

**Section 1: Soil Nutrients**
- Label: "Soil Nutrients"
- Layout: 3-column grid (responsive to 1 column on mobile)
- Fields:
  1. **Nitrogen (N)** - Range: 0-140
  2. **Phosphorus (P)** - Range: 5-145
  3. **Potassium (K)** - Range: 5-205

**Section 2: Environmental Conditions**
- Label: "Environmental Conditions"
- Layout: 2×2 grid (responsive to 1 column on mobile)
- Fields:
  1. **Temperature** - Range: 8.8-43.7°C, Unit: °C
  2. **Humidity** - Range: 14.3-99.9%, Unit: %
  3. **pH** - Range: 3.5-9.9
  4. **Rainfall** - Range: 20.2-298.6mm, Unit: mm

#### Input Field Styling
- **Background**: Dark card background (`hsl(220, 15%, 12%)`)
- **Border**: 2px solid subtle border
- **Border Radius**: `0.5rem` (--radius-md)
- **Padding**: `1rem 1.5rem`
- **Font Size**: `1rem`
- **Floating Label**: Label floats up when field has value or is focused

**States**:
- **Default**: Subtle border
- **Focus**: Green border with glow (`box-shadow: 0 0 0 4px var(--primary-glow)`)
- **Valid**: Green border with checkmark icon
- **Invalid**: Orange border with shake animation and error message

#### Submit Button
- **Text**: "Get Recommendation"
- **Style**: Primary button with gradient and glow
- **Width**: Full width
- **Margin Top**: `2rem`
- **Loading State**: Spinner replaces text during prediction

### Implementation Notes

```html
<div class="prediction-container container">
  <div class="prediction-card card-glass">
    <h2 class="form-title">🌾 Crop Recommendation Input</h2>
    
    <form id="predictionForm" class="prediction-form">
      <div class="form-section">
        <h3 class="section-title">Soil Nutrients</h3>
        <div class="form-grid grid-3">
          <div class="form-group">
            <label for="nitrogen" class="floating-label">Nitrogen (N)</label>
            <input type="number" id="nitrogen" name="N" 
                   class="input-field" min="0" max="140" required>
            <span class="input-hint">0-140</span>
          </div>
          <!-- Repeat for P, K -->
        </div>
      </div>
      
      <div class="form-section">
        <h3 class="section-title">Environmental Conditions</h3>
        <div class="form-grid grid-2">
          <!-- Temperature, Humidity, pH, Rainfall fields -->
        </div>
      </div>
      
      <button type="submit" class="btn btn-primary btn-full">
        Get Recommendation
      </button>
    </form>
  </div>
</div>
```

---

## 3️⃣ Results Card (`03_results_card.png`)

### Description
Displays the crop recommendation result with confidence score, reasoning, and alternative suggestions. Appears with a slide-up animation after form submission.

### Key Elements

#### Primary Result Card
- **Layout**: Large, prominent card with gradient border
- **Background**: Glassmorphism with subtle green glow
- **Border**: Animated gradient border on load

**Content Structure**:
1. **Header**: "🎯 Recommended Crop"
2. **Crop Icon/Image**: Large illustration of the recommended crop
3. **Crop Name**: Large text with gradient effect (e.g., "RICE")
4. **Confidence Score**: 
   - Large percentage (e.g., "98.5%")
   - Animated progress bar
   - Color: Green gradient
5. **Reasoning Section**: "Why this crop?"
   - Bullet points explaining the recommendation
   - Examples:
     - "✓ Optimal nitrogen levels"
     - "✓ Perfect humidity range"
     - "✓ Suitable rainfall"

**Special Effects**:
- If confidence > 95%: Confetti animation
- Progress bar animates from 0 to final value
- Card pulses with subtle glow

#### Alternative Crops Section
- **Layout**: 3 smaller cards in a row (responsive to stack on mobile)
- **Cards**: Show top 3 alternative crops
- **Content per card**:
  - Crop name
  - Confidence percentage
  - Small crop icon

**Card Styling**:
- Background: Glassmorphism
- Border: Subtle border
- Hover: Expand slightly with details
- Click: Show comparison with main recommendation

#### Visualization Section
Below the results, two interactive charts:

1. **Feature Importance Bar Chart**
   - Shows top 5 features influencing the decision
   - Horizontal bars with gradient colors
   - Animated growth from left to right
   - Hover: Show exact values

2. **Suitability Radar Chart**
   - 6-axis radar: N, P, K, Temperature, Humidity, Rainfall
   - Overlay: Ideal range vs. input values
   - Animated drawing from center
   - Interactive: Hover to see specific values

### Implementation Notes

```html
<div class="results-container container">
  <!-- Primary Result -->
  <div class="result-card card-glass">
    <h2 class="result-header">🎯 Recommended Crop</h2>
    
    <div class="crop-icon">
      <img src="/static/images/crops/rice.png" alt="Rice">
    </div>
    
    <h3 class="crop-name text-gradient">RICE</h3>
    
    <div class="confidence-section">
      <div class="confidence-value">98.5%</div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width: 98.5%"></div>
      </div>
    </div>
    
    <div class="reasoning-section">
      <h4>Why this crop?</h4>
      <ul class="reason-list">
        <li>✓ Optimal nitrogen levels</li>
        <li>✓ Perfect humidity range</li>
        <li>✓ Suitable rainfall</li>
      </ul>
    </div>
  </div>
  
  <!-- Alternative Crops -->
  <div class="alternatives-section">
    <h3>Alternative Recommendations</h3>
    <div class="alternatives-grid">
      <div class="alternative-card card-glass">
        <div class="alt-crop-name">Wheat</div>
        <div class="alt-confidence">92.3%</div>
      </div>
      <!-- Repeat for other alternatives -->
    </div>
  </div>
  
  <!-- Charts -->
  <div class="charts-section">
    <div class="chart-container">
      <canvas id="featureImportanceChart"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="suitabilityRadarChart"></canvas>
    </div>
  </div>
</div>
```

---

## 4️⃣ Performance Dashboard (`04_performance_dashboard.png`)

### Description
Displays comprehensive model performance metrics in an organized, visually appealing dashboard layout.

### Key Elements

#### Metrics Grid (Top Section)
- **Layout**: 2×2 grid (responsive to 1 column on mobile)
- **Cards**: 4 glassmorphism metric cards

**Card 1: Test Accuracy**
- Icon: Target/bullseye
- Value: "99.32%"
- Label: "Test Accuracy"
- Animation: Count-up from 0 to 99.32

**Card 2: Training Time**
- Icon: Clock
- Value: "0.82s"
- Label: "Training Time"
- Animation: Count-up with time format

**Card 3: Prediction Time**
- Icon: Lightning bolt
- Value: "8.56ms"
- Label: "Prediction Time"
- Animation: Count-up with ms format

**Card 4: Model Size**
- Icon: File/document
- Value: "4.76 MB"
- Label: "Model Size"
- Animation: Count-up with MB format

**Metric Card Styling**:
- Background: Glassmorphism with gradient
- Icon: Large, centered at top
- Value: Extra large font with gradient
- Label: Smaller, secondary color
- Hover: Glow effect and slight lift

#### Model Comparison Table
- **Layout**: Side-by-side comparison
- **Header**: "Random Forest vs XGBoost"
- **Columns**: Metric | Random Forest | XGBoost | Winner

**Metrics Compared**:
1. Accuracy
2. Training Speed
3. Inference Speed
4. Model Size
5. Cross-Val Stability

**Styling**:
- Winning metrics: Green highlight with glow
- Table rows: Alternate subtle background
- Hover: Row highlights
- Toggle: Switch between table and chart view

#### Additional Visualizations
Below the comparison, links or thumbnails to:
- Confusion matrices
- Feature importance charts
- ROC curves
- Cross-validation plots

### Implementation Notes

```html
<div class="performance-container container">
  <h1 class="page-title">Model Performance</h1>
  
  <!-- Metrics Grid -->
  <div class="metrics-grid">
    <div class="metric-card card-glass">
      <div class="metric-icon">🎯</div>
      <div class="metric-value" data-target="99.32">0</div>
      <div class="metric-unit">%</div>
      <div class="metric-label">Test Accuracy</div>
    </div>
    <!-- Repeat for other metrics -->
  </div>
  
  <!-- Comparison Section -->
  <div class="comparison-section">
    <h2>Model Comparison</h2>
    <div class="comparison-toggle">
      <button class="toggle-btn active" data-view="table">Table</button>
      <button class="toggle-btn" data-view="chart">Chart</button>
    </div>
    
    <table class="comparison-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Random Forest</th>
          <th>XGBoost</th>
          <th>Winner</th>
        </tr>
      </thead>
      <tbody>
        <tr class="winner-rf">
          <td>Test Accuracy</td>
          <td>99.32%</td>
          <td>99.09%</td>
          <td>✓</td>
        </tr>
        <!-- More rows -->
      </tbody>
    </table>
  </div>
</div>
```

---

## 5️⃣ Features Section (`05_features_section.png`)

### Description
A grid of feature cards showcasing the key benefits and capabilities of the crop recommendation system.

### Key Elements

#### Section Header
- **Title**: "Why Choose Our System?"
- **Subtitle**: "Powered by advanced machine learning and rigorous testing"
- **Alignment**: Center
- **Spacing**: Large margin bottom

#### Features Grid
- **Layout**: 3-column grid (responsive: 2 columns on tablet, 1 on mobile)
- **Gap**: `2rem` between cards
- **Cards**: 6 feature cards total

**Feature Cards**:

1. **High Accuracy**
   - Icon: 🎯 Target
   - Title: "High Accuracy"
   - Description: "99.3% accuracy with Random Forest model"
   - Stat: "99.3%"

2. **Lightning Fast**
   - Icon: ⚡ Lightning bolt
   - Title: "Lightning Fast"
   - Description: "Get predictions in under 10ms"
   - Stat: "<10ms"

3. **Scientifically Validated**
   - Icon: 🧪 Flask/beaker
   - Title: "Scientifically Validated"
   - Description: "Based on soil nutrients and environmental data"
   - Stat: "7 Features"

4. **Data-Driven Insights**
   - Icon: 📊 Chart/graph
   - Title: "Data-Driven Insights"
   - Description: "Comprehensive analysis with visualizations"
   - Stat: "22 Features"

5. **22 Crop Types**
   - Icon: 🌱 Seedling
   - Title: "22 Crop Types"
   - Description: "From rice to watermelon, we've got you covered"
   - Stat: "22 Crops"

6. **Rigorously Tested**
   - Icon: 🔬 Microscope
   - Title: "Rigorously Tested"
   - Description: "200+ tests ensure reliability"
   - Stat: "200+ Tests"

**Card Styling**:
- Background: Glassmorphism with subtle gradient
- Border: Subtle white border
- Border Radius: `1rem`
- Padding: `2rem`
- Icon: Large (3-4rem), centered at top
- Title: Bold, medium size
- Description: Secondary color, smaller
- Stat: Large, gradient text at bottom

**Animations**:
- On scroll: Staggered fade-in (each card delays by 100ms)
- On hover: 
  - Card lifts up (`translateY(-8px)`)
  - Shadow increases
  - Border glows with green color
  - Icon animates (specific to each icon type)

### Implementation Notes

```html
<section class="features-section">
  <div class="container">
    <h2 class="section-title">Why Choose Our System?</h2>
    <p class="section-subtitle">
      Powered by advanced machine learning and rigorous testing
    </p>
    
    <div class="features-grid">
      <div class="feature-card card-glass" data-aos="fade-up" data-aos-delay="0">
        <div class="feature-icon">🎯</div>
        <h3 class="feature-title">High Accuracy</h3>
        <p class="feature-description">
          99.3% accuracy with Random Forest model
        </p>
        <div class="feature-stat text-gradient">99.3%</div>
      </div>
      
      <div class="feature-card card-glass" data-aos="fade-up" data-aos-delay="100">
        <div class="feature-icon">⚡</div>
        <h3 class="feature-title">Lightning Fast</h3>
        <p class="feature-description">
          Get predictions in under 10ms
        </p>
        <div class="feature-stat text-gradient">&lt;10ms</div>
      </div>
      
      <!-- Repeat for other features -->
    </div>
  </div>
</section>
```

---

## 🎨 Common Design Patterns Across Mockups

### Glassmorphism Effect
```css
.card-glass {
  background: linear-gradient(
    135deg,
    hsla(220, 15%, 12%, 0.7) 0%,
    hsla(220, 15%, 15%, 0.5) 100%
  );
  backdrop-filter: blur(20px);
  border: 1px solid hsla(0, 0%, 100%, 0.1);
  border-radius: 1rem;
}
```

### Gradient Text
```css
.text-gradient {
  background: linear-gradient(
    135deg,
    hsl(142, 71%, 45%) 0%,
    hsl(210, 100%, 56%) 100%
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

### Hover Lift Effect
```css
.card-glass:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  border-color: var(--primary-green);
}
```

### Glow Effect
```css
.glow {
  box-shadow: 0 0 20px hsla(142, 71%, 45%, 0.3);
}

.glow:hover {
  box-shadow: 0 0 40px hsla(142, 71%, 45%, 0.5);
}
```

---

## 📱 Responsive Considerations

### Breakpoints Used in Mockups
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Mobile Adaptations
1. **Hero Section**: Stack buttons vertically, reduce heading size
2. **Prediction Form**: Single column layout for all inputs
3. **Results**: Stack alternative crops vertically
4. **Performance Dashboard**: Single column for metric cards
5. **Features Section**: Single column card layout

---

## 🎯 Implementation Priority

### Phase 1 (Essential)
1. Hero Section - Sets the tone
2. Prediction Form - Core functionality
3. Results Card - Shows value

### Phase 2 (Important)
4. Features Section - Builds trust
5. Performance Dashboard - Demonstrates quality

### Phase 3 (Enhancement)
- Add animations
- Optimize for mobile
- Polish interactions

---

## 📚 Additional Resources

### Fonts
- **Inter**: https://fonts.google.com/specimen/Inter
- **Outfit**: https://fonts.google.com/specimen/Outfit
- **JetBrains Mono**: https://fonts.google.com/specimen/JetBrains+Mono

### Icons
- **Lucide Icons**: https://lucide.dev/
- **Heroicons**: https://heroicons.com/

### Charts
- **Chart.js**: https://www.chartjs.org/
- **ApexCharts**: https://apexcharts.com/

### Animations
- **AOS (Animate On Scroll)**: https://michalsnik.github.io/aos/
- **GSAP**: https://greensock.com/gsap/

---

**Note**: All mockups are saved in `design/mockups/` and can be referenced during implementation. Use these as visual guides while following the detailed specifications in `UI_DESIGN.md`.
